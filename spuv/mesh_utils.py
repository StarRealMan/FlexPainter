import os
import numpy as np
import torch
from PIL import Image
import time
import OpenEXR
import Imath
import torch.nn.functional as F
import trimesh
import torch.nn.functional as F

def load_mesh_and_pbr(
        mesh_path,
        diffuse_path,
):
    mesh = trimesh.load(mesh_path, process=False)
    # Load texture and mask, flip vertically
    diffuse_map = torch.from_numpy(load_image(diffuse_path))
    diffuse_map = torch.flip(diffuse_map, dims=[0])

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    uv_vertices = torch.tensor(mesh.visual.uv, dtype=torch.float32)
    uv_faces = torch.tensor(mesh.faces, dtype=torch.int64)
    v_nrm = _compute_vertex_normal(faces, vertices)

    mesh = {
        'v_pos': vertices,
        't_pos_idx': faces,
        '_v_tex': uv_vertices,
        '_t_tex_idx': uv_faces,
        'v_nrm': v_nrm,
    }

    return mesh, diffuse_map

def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def _compute_vertex_normal(t_pos_idx, v_pos):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm

def padding_iter(image, mask, window_size=3):
    C, H, W = image.shape
    pad_size = window_size // 2
    padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    padded_mask = F.pad(mask, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    image_patches = padded_image.unfold(1, window_size, 1).unfold(2, window_size, 1)
    mask_patches = padded_mask.unfold(1, window_size, 1).unfold(2, window_size, 1)

    # Reshape the mask to merge the last two dimensions
    mask_patches_flat = mask_patches.reshape(-1, H, W, window_size * window_size)

    # Find the indices of the first occurrence of 1 in the flattened mask
    mask_patches_flat_int = (mask_patches_flat == 1).type(torch.int)
    indices = torch.argmax(mask_patches_flat_int, dim=-1)

    # Convert flat indices back to 3x3 patch indices
    di_indices = indices // window_size
    dj_indices = indices % window_size

    # Gather the new image patches based on the calculated indices
    new_image = image_patches[torch.arange(image_patches.shape[0])[:, None, None],
                            torch.arange(H)[None, :, None],
                            torch.arange(W)[None, None, :],
                            di_indices, dj_indices]

    updated_image = torch.where(mask == 1, image, new_image)

    # Create a condition tensor where indices are not zero
    edge_condition = (indices != 0)

    # Update the mask: set to 1 where condition is True
    # Ensure that the mask is of the same dtype as condition (e.g., int or bool)
    updated_mask = torch.where(edge_condition, torch.ones_like(mask), mask)

    return updated_image, updated_mask

def uv_padding(image, mask, window_size=3, iterations=8):
    image = image.permute(2, 0, 1)
    mask = mask.unsqueeze(0)

    updated_image, updated_mask = image.clone(), mask.clone()
    for _ in range(iterations):
        updated_image, updated_mask = padding_iter(updated_image, updated_mask, window_size)

    blended_image = torch.where(mask == 1, image, updated_image)

    blended_image = blended_image.permute(1, 2, 0).unsqueeze(0)
    return blended_image


class TimeRecorder:
    def __init__(self):
        self.records = {}

    def start(self, task_name):
        """Start a new timing record for the given task."""
        self.records[task_name] = {'start': time.time(), 'duration': None}

    def end(self, task_name):
        """End the timing record for the given task and calculate the duration."""
        if task_name in self.records and 'start' in self.records[task_name]:
            end_time = time.time()
            start_time = self.records[task_name]['start']
            self.records[task_name]['duration'] = end_time - start_time
        else:
            print(f"Task '{task_name}' was not started or is already ended.")

    def get_duration(self, task_name):
        """Retrieve the duration of a specific task."""
        return self.records.get(task_name, {}).get('duration')

    def print_durations(self):
        """Prints the durations of all recorded tasks."""
        for task, record in self.records.items():
            if record['duration'] is not None:
                print(f"{task}: {record['duration']} seconds")
            else:
                print(f"{task}: not completed or not started")

    def print_all_durations(self):
        """Prints the duration for all recorded tasks."""
        for task, record in self.records.items():
            duration = record.get('duration')
            if duration is not None:
                print(f"{task}: {duration:.2f} seconds")
            else:
                print(f"{task}: not completed or not started")


def convert_to_save_format(mesh_vertices, mesh_faces, texture_uvs):
    points = mesh_vertices
    faces = mesh_faces

    # Flatten texture_uvs and convert to a 2D numpy array
    flattened_uvs = texture_uvs.reshape(-1, 2)

    # Find unique UV coordinates and their indices
    unique_uvs, inverse_indices = np.unique(flattened_uvs, axis=0, return_inverse=True)

    # Reconstruct texture_faces array using inverse_indices
    texture_faces = inverse_indices.reshape(-1, 3)

    return points, unique_uvs, faces, texture_faces

def write_obj_with_uv(mesh_vertices, mesh_faces, texture_uvs, obj_file_path):
    """
    Writes mesh data to an OBJ file including UV coordinates.

    :param mesh_vertices: (N, 3) array of vertices
    :param mesh_faces: (F, 3) array of face indices
    :param texture_uvs: (F, 3, 2) array of UV coordinates for each face's vertices
    :param obj_file_path: Path to save the OBJ file
    """
    with open(obj_file_path, 'w') as file:
        # Write vertices
        for vertex in mesh_vertices:
            file.write('v {} {} {}\n'.format(*vertex))

        # Write UVs and create a mapping from UVs to their indices
        uv_indices = {}
        uv_counter = 1
        for face_uv in texture_uvs:
            for uv in face_uv:
                uv_tuple = tuple(uv)
                if uv_tuple not in uv_indices:
                    file.write('vt {} {}\n'.format(*uv))
                    uv_indices[uv_tuple] = uv_counter
                    uv_counter += 1

        # Write faces with vertex and UV indices
        for face_idx, face_uv in zip(mesh_faces, texture_uvs):
            face_line = 'f'
            for v_idx, uv in zip(face_idx, face_uv):
                # OBJ file indices are 1-based, so add 1 to each index
                face_line += ' {}/{}'.format(v_idx + 1, uv_indices[tuple(uv)])
            face_line += '\n'
            file.write(face_line)

def save_mesh_to_file(points, texture_coords, faces, texture_faces, file_name):
    folder, name = os.path.split(file_name)
    name, _ = os.path.splitext(name)

    material_file_name = f'{folder}/{name}.mtl'
    with open(material_file_name, 'w') as mat_file:
        mat_file.write('newmtl material_0\nKd 1 1 1\nKa 0 0 0\nKs 0.4 0.4 0.4\nNs 10\nillum 2\nmap_Kd {name}.png\n')

    with open(file_name, 'w') as obj_file:
        obj_file.write(f'mtllib {name}.mtl\n')

        for point in points:
            obj_file.write(f'v {point[0]} {point[1]} {point[2]}\n')

        for coord in texture_coords:
            obj_file.write(f'vt {coord[0]} {coord[1]}\n')

        obj_file.write('usemtl material_0\n')
        for face, tex_face in zip(faces, texture_faces):
            face = face + 1  # Adjusting for 1-based indexing in OBJ format
            tex_face = tex_face + 1
            obj_file.write(f'f {face[0]}/{tex_face[0]} {face[1]}/{tex_face[1]} {face[2]}/{tex_face[2]}\n')

def save_ply(points, normals, filename):
    points_count = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "end_header\n"
    )
    with open(filename, 'w') as ply_file:
        ply_file.write(header.format(points_count))
        for i in range(points_count):
            ply_file.write("{} {} {} {} {} {}\n".format(
                points[i, 0], points[i, 1], points[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2]
            ))

def save_as_exr(array, path):
    """
    Save a high precision image with a mask as an EXR image.

    :param array: The input array, assumed to be in float16.
    :param path: The path to save the image.
    """

    # Convert to float32
    rgb_image_32bit = array.astype(np.float32)

    # Create EXR file
    exr_file = OpenEXR.OutputFile(path, OpenEXR.Header(rgb_image_32bit.shape[1], rgb_image_32bit.shape[0]))

    # Write to EXR file
    exr_data = [rgb_image_32bit[:, :, i].tobytes() for i in range(3)]  # Convert each channel to bytes

    exr_file.writePixels({'R': exr_data[0], 'G': exr_data[1], 'B': exr_data[2]})
    exr_file.close()

def save_depth_as_exr(depth_array, path):
    """
    Save depth information as an EXR image.

    :param depth_array: The input depth array, assumed to be in float16 or float32.
    :param path: The path to save the depth image.
    """

    depth_array_32bit = depth_array.astype(np.float32)

    header = OpenEXR.Header(depth_array_32bit.shape[1], depth_array_32bit.shape[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'Y': half_chan}

    exr_file = OpenEXR.OutputFile(path, header)
    depth_data = depth_array_32bit.tobytes()  

    exr_file.writePixels({'Y': depth_data})
    exr_file.close()

def load_exr_as_rgb(path):
    """
    Load an EXR image and return the RGB image and mask.

    :param path: Path to the EXR image file.
    :return: RGB image.
    """
    exr_file = OpenEXR.InputFile(path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    def read_channel(channel):
        return np.frombuffer(exr_file.channel(channel, FLOAT), dtype=np.float32).reshape(size[1], size[0])

    r, g, b = [read_channel(c) for c in ('R', 'G', 'B')]
    rgb = np.stack((r, g, b), axis=-1)

    return rgb

def save_unnormalize_image_with_mask(array, mask, path):
    """
    Save an unnormalized image with a mask as an RGBA image.

    :param array: The input array, assumed to be normalized in the range [-1, 1].
    :param mask: The mask array, with either one or more channels.
    :param path: The path to save the image.
    """
    # Normalize the input array to [0, 255] and convert to uint8
    normalized_array = (((array + 1) * 0.5) * 255).astype(np.uint8)

    # If the mask has more than one channel, use the first channel
    if mask.ndim > 2:
        mask = mask[..., 0]
    # Invert and normalize the mask to [0, 255] and convert to uint8
    normalized_mask = ( mask * 255).astype(np.uint8)

    # Create an RGBA image by combining the array and the mask
    rgba_image = np.concatenate((normalized_array, normalized_mask[..., np.newaxis]), axis=-1)

    # Save the RGBA image
    Image.fromarray(rgba_image).save(path)

def transform_for_blender(vertices):
    """Transform coordinates to match Blender's default setup where -y is forward and z is up."""
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    return np.dot(vertices, rotation_matrix)

def preprocess_mesh(mesh, mesh_scale=1.0, rotate_to_blender=False):
    vertices = mesh.vertices

    # Rotate vertices for Blender compatibility, if required.
    if rotate_to_blender:
        vertices = transform_for_blender(vertices)

    # Scale and center the mesh vertices.
    # The scaling is determined based on the largest dimension to maintain aspect ratio.
    # Centering is done by aligning the mesh center to the origin (0,0,0).
    bounding_box_max = vertices.max(0)
    bounding_box_min = vertices.min(0)
    scale = mesh_scale / (bounding_box_max - bounding_box_min).max()
    center_offset = (bounding_box_max + bounding_box_min) * 0.5
    vertices = (vertices - center_offset) * scale

    mesh.vertices = vertices

    return mesh

def load_image(path):
    if path.endswith('.png') or path.endswith('.jpg'):
        image = Image.open(path)
        image = np.array(image) / 255
    elif path.endswith('.exr'):
        image = load_exr_as_rgb(path)
    else:
        raise ValueError(f"Unsupported image format: {path}")

    return image

def load_mesh_only(mesh_path, device):
    """
    Loads a mesh and its corresponding texture and mask from files.

    :param mesh_path: Path to the mesh file (OBJ format).
    :param img_path: Path to the texture image.
    :return: A tuple containing the mesh data, texture map, and mask map.
    """

    # Load texture and mask, flip vertically

    mesh = trimesh.load(mesh_path, process=False, force='mesh')
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    uv_vertices = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
    uv_faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    v_norm = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device).contiguous()

    # Create a dictionary to hold mesh data
    mesh = {
        'v_pos': vertices,
        't_pos_idx': faces,
        'v_norm': v_norm,
        '_v_tex': uv_vertices,
        '_t_tex_idx': uv_faces
    }

    return mesh

def load_mesh_and_uv(mesh_path, img_path, device):
    """
    Loads a mesh and its corresponding texture and mask from files.

    :param mesh_path: Path to the mesh file (OBJ format).
    :param img_path: Path to the texture image.
    :return: A tuple containing the mesh data, texture map, and mask map.
    """

    # Load texture and mask, flip vertically
    tex_map = torch.from_numpy(load_image(img_path)).to(device)
    tex_map = torch.flip(tex_map, dims=[0])

    mesh = trimesh.load(mesh_path, process=False, force='mesh')
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    uv_vertices = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
    uv_faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    v_norm = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device).contiguous()

    # Create a dictionary to hold mesh data
    mesh = {
        'v_pos': vertices,
        't_pos_idx': faces,
        'v_norm': v_norm,
        '_v_tex': uv_vertices,
        '_t_tex_idx': uv_faces
    }

    return mesh, tex_map

def load_mesh_and_uv_and_texture(mesh_path, position_path, img_path, mask_path, device):
    """
    Loads a mesh and its corresponding texture and mask from files.

    :param mesh_path: Path to the mesh file (OBJ format).
    :param img_path: Path to the texture image.
    :param mask_path: Path to the mask image.
    :param device: The device to which the tensors are sent (e.g., 'cuda').
    :return: A tuple containing the mesh data, texture map, and mask map.
    """

    # Load texture and mask, flip vertically
    tex_map = torch.from_numpy(load_image(img_path)).to(device)
    position_map = torch.from_numpy(load_image(position_path)).to(device)
    mask_map = torch.from_numpy(load_image(mask_path)).to(device)
    tex_map = torch.flip(tex_map, dims=[0])
    position_map = torch.flip(position_map, dims=[0])
    mask_map = torch.flip(mask_map, dims=[0])

    # Initialize lists to store mesh data
    vertex_data, face_data, uv_vertex_data, uv_face_data = [], [], [], []

    # Process the OBJ file
    with open(mesh_path, "r") as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue

            values = line.split()
            if values[0] == 'v':
                # Process vertex coordinates
                v = list(map(float, values[1:4]))
                vertex_data.append(v)
            elif values[0] == 'vt':
                # Process UV coordinates
                vt = list(map(float, values[1:3]))
                uv_vertex_data.append(vt)
            elif values[0] == 'f':
                # Process face indices (vertices and UVs)
                f = [int(x.split('/')[0]) for x in values[1:4]]
                uv_f = [int(x.split('/')[1]) for x in values[1:4]]
                face_data.append(f)
                uv_face_data.append(uv_f)

    # Convert lists to tensors and adjust for 1-based indexing
    vertices = torch.tensor(vertex_data, dtype=torch.float32, device=device)
    faces = torch.tensor(face_data, dtype=torch.int32, device=device) - 1
    uv_vertices = torch.tensor(uv_vertex_data, dtype=torch.float32, device=device)
    uv_faces = torch.tensor(uv_face_data, dtype=torch.int32, device=device) - 1

    # Create a dictionary to hold mesh data
    mesh = {
        'v_pos': vertices,
        't_pos_idx': faces,
        '_v_tex': uv_vertices,
        '_t_tex_idx': uv_faces
    }

    return mesh, position_map, tex_map, mask_map

def load_mesh_from_file(file_name):
    points = []
    texture_coords = []
    faces = []
    texture_faces = []

    with open(file_name, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):  # Vertex coordinates
                _, x, y, z = line.split()
                points.append((float(x), float(y), float(z)))
            elif line.startswith('vt '):  # Texture coordinates
                _, u, v = line.split()
                texture_coords.append((float(u), float(v)))
            elif line.startswith('f '):  # Face definitions
                _, v1, v2, v3 = line.split()
                face = [int(vertex.split('/')[0]) for vertex in [v1, v2, v3]]
                tex_face = [int(vertex.split('/')[1]) for vertex in [v1, v2, v3]]
                faces.append(tuple([f - 1 for f in face]))  # Adjust for 0-based indexing
                texture_faces.append(tuple([tf - 1 for tf in tex_face]))

    # Convert lists to numpy arrays
    points = np.array(points)
    texture_coords = np.array(texture_coords)
    faces = np.array(faces)
    texture_faces = np.array(texture_faces)

    return points, texture_coords, faces, texture_faces

def vertex_transform(mesh, mesh_scale=1.0):
    pre_vertices = mesh['v_pos']

    vertices = torch.clone(pre_vertices)
    vertices[:, 1] = -pre_vertices[:, 2]  # -z --> y
    vertices[:, 2] = pre_vertices[:, 1]  # y --> z

    bounding_box_max = vertices.max(0)[0]
    bounding_box_min = vertices.min(0)[0]
    scale = 2 * mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6)
    center_offset = (bounding_box_max + bounding_box_min) * 0.5
    vertices = (vertices - center_offset) * scale

    mesh['v_pos'] = vertices

    return mesh