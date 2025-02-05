import bpy
import os
import math
import matplotlib.colors
import numpy as np
import logging
import matplotlib
import uuid


COUNT = 15
base_dir = "D:/RESEARCH/Code/MDM/mdm/results/"
results_1_0 = "1/sample00_rep00_obj/"
results_1_1 = "1/sample00_rep01_obj/"
results_1_2 = "1/sample00_rep02_obj/"
results_1_3 = "1/sample00_rep03_obj/"
results_1_4 = "1/sample00_rep04_obj/"

results_2_0 = "2/sample00_rep00_obj/"
results_2_1 = "2/sample00_rep01_obj/"
results_2_7 = "2/sample00_rep07_obj/"
results_2_8 = "2/sample00_rep08_obj/"

results_3_0 = "3/sample00_rep00_obj/"
results_3_1 = "3/sample00_rep01_obj/"
results_3_4 = "3/sample00_rep04_obj/"
results_3_9 = "3/sample00_rep09_obj/"

results_4_1 = "4/sample00_rep01_obj/"
results_4_2 = "4/sample00_rep02_obj/"
results_4_5 = "4/sample00_rep05_obj/"
results_4_9 = "4/sample00_rep09_obj/"

ours_1_0 = "ours_1/sample00_rep00_obj/"
ours_1_2 = "ours_1/sample00_rep02_obj/"
ours_1_3 = "ours_1/sample00_rep03_obj/"

novice_kick_ball = "novice_kicking_a_ball/sample00_rep00_obj/"
professional_kick_ball = "professional_kicking_a_ball/sample00_rep03_obj/"
arm_streched_legs_bent_slightly = "arm_streched_legs_bent_slightly/sample00_rep00_obj/"
balancing_on_a_rope = "balancing_on_a_rope/sample00_rep03_obj/"
crawling_in_a_circle = "crawling_in_a_circle/sample00_rep03_obj/"
jump_rope = "jump_rope/sample00_rep03_obj/"
flip_roll = "flip_roll/sample00_rep01_obj/"
hopping = "ours_1/sample00_rep02_obj/"

test = "arm_bent/"

file = test
obj_directory = base_dir + file






color_background = "#FFFFFF"
color_smpl = "#FFA800"
color_plane1 = "#444444"
color_plane2 = "#121212"

smpl_opacity_min = 0
smpl_opacity_max = 100


def hex_to_rgb(hex_color):
    rgb = list(matplotlib.colors.to_rgb(hex_color))
    rgb.append(1.0)
    return tuple(rgb)


def hsv_to_rgb(h, s, v):
    rgb = list(matplotlib.colors.hsv_to_rgb((h, s, v)))
    rgb.append(1.0)
    return tuple(rgb)


def get_frame_indices(num_frames, count):
    lst = list(range(0, num_frames))  # Generate list of numbers from 1 to num_items
    if count <= 1:
        return [lst[0], lst[-1]]  # If count is 1, return the first and last items only
    step = (len(lst) - 1) / (count - 1)
    selected_numbers = [lst[0]]  # Include the first element
    for i in range(1, count - 1):  # Select the intermediate elements
        selected_numbers.append(lst[int(i * step)])
    selected_numbers.append(lst[-1])  # Include the last element
    return selected_numbers




def get_alphas(count, min=70, max=100):
    if count <= 1:
        return [lst[0], lst[-1]]
    
    return [i/100.0 for i in np.linspace(min, max, count)]


def get_smpl_frame_color(frame, nframes, start=0.3, end=1.0):        
    cmap = matplotlib.cm.get_cmap('Blues')
    rgbcolor = cmap(start + (end-start)*(frame/nframes))
    return rgbcolor












def setup_renderer():

    # Other common render settings
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPTIX'

    bpy.context.scene.cycles.use_adaptive_sampling = True
    
    bpy.context.scene.cycles.samples = 4098
    
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.gamma = 1.2
    bpy.context.scene.view_settings.exposure = -0.75



def setup_scene(res="ultra"):
    scene = bpy.data.scenes['Scene']
    assert res in ["ultra", "high", "med", "low"]

    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280//2
        scene.render.resolution_y = 1024//2
    elif res == "low":
        scene.render.resolution_x = 1280//4
        scene.render.resolution_y = 1024//4
    elif res == "ultra":
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080

    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
        
    if not 'Sun' in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', align='WORLD',
                                 location=(0, 0, 0), scale=(1, 1, 1))
        bpy.data.objects["Sun"].data.energy = 5

    if not 'Empty' in bpy.data.objects:
    # rotate camera
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                 orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1,
                                 use_proportional_connected=False, use_proportional_projected=False)
    
    bpy.ops.object.select_all(action='DESELECT')

    return scene


def plot_floor():
    # Create a floor
    location_small = (0,0,0)
    location_big = (0,0,-0.01)
    scale_small = (1,1,1)
    scale_big = (1.2,1.2,1.2)
    
    color_small = (0.1, 0.1, 0.1, 1)
    color_big = (0.2, 0.2, 0.2, 1)

    if not 'SmallPlane' in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location_small, scale=scale_small)

        bpy.ops.transform.resize(value=scale_small, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)
        obj = bpy.context.selected_objects[0]
        obj.name = "SmallPlane"
        obj.data.name = "SmallPlane"
        obj.active_material = floor_mat(color=color_small)


    if not 'BigPlane' in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location_big, scale=scale_big)

        bpy.ops.transform.resize(value=scale_big, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                     constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                     proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                     use_proportional_projected=False, release_confirm=True)

        obj = bpy.context.selected_objects[0]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=color_big)

def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material_diffuse_BSDF(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material

# keys:
# ['Base Color', 'Subsurface', 'Subsurface Radius', 'Subsurface Color', 'Metallic', 'Specular', 'Specular Tint', 'Roughness', 'Anisotropic', 'Anisotropic Rotation', 'Sheen', 1Sheen Tint', 'Clearcoat', 'Clearcoat Roughness', 'IOR', 'Transmission', 'Transmission Roughness', 'Emission', 'Emission Strength', 'Alpha', 'Normal', 'Clearcoat Normal', 'Tangent']
DEFAULT_BSDF_SETTINGS = {"Subsurface": 0.15,
                         "Subsurface Radius": [1.1, 0.2, 0.1],
                         "Metallic": 0.3,
                         "Specular": 0.5,
                         "Specular Tint": 0.5,
                         "Roughness": 0.75,
                         "Anisotropic": 0.25,
                         "Anisotropic Rotation": 0.25,
                         "Sheen": 0.75,
                         "Sheen Tint": 0.5,
                         "Clearcoat": 0.5,
                         "Clearcoat Roughness": 0.5,
                         "IOR": 1.450,
                         "Transmission": 0.1,
                         "Transmission Roughness": 0.1,
                         "Emission": (0, 0, 0, 1),
                         "Emission Strength": 0.0,
                         "Alpha": 1.0}

def body_material(r, g, b, a=1, name="body"):
    materials = bpy.data.materials
    material = materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    diffuse = nodes["Principled BSDF"]
    inputs = diffuse.inputs

    settings = DEFAULT_BSDF_SETTINGS.copy()
    settings["Base Color"] = (r, g, b, a)
    settings["Subsurface Color"] = (r, g, b, a)
    settings["Subsurface"] = 0.0

    for setting, val in settings.items():
        inputs[setting].default_value = val

    return material


def colored_material_bsdf(name, **kwargs):
    materials = bpy.data.materials
    material = materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    diffuse = nodes["Principled BSDF"]
    inputs = diffuse.inputs

    settings = DEFAULT_BSDF_SETTINGS.copy()
    for key, val in kwargs.items():
        settings[key] = val

    for setting, val in settings.items():
        inputs[setting].default_value = val

    return material


def floor_mat(name="floor_mat", color=(0.1, 0.1, 0.1, 1), roughness=0.127451):
    return colored_material_diffuse_BSDF(color[0], color[1], color[2], a=color[3], roughness=roughness)




setup_renderer()
setup_scene()
plot_floor()









obj_files = sorted([f for f in os.listdir(obj_directory) if f.endswith(".obj")])
num_frames = len(obj_files)

frame_indices = get_frame_indices(num_frames, COUNT) 
alphas = get_alphas(COUNT, min=smpl_opacity_min, max=smpl_opacity_max)

obj_files = [obj_files[i] for i in frame_indices]



        

# List to keep track of all smpls
smpls = []

# Loop through all OBJ files and import them
for index, obj_file in enumerate(obj_files):
    # Import the OBJ file using wm.obj_import
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_directory, obj_file))

    # Get the newly imported object
    smpl = bpy.context.selected_objects[0]
    smpls.append(smpl)
    bpy.ops.object.shade_smooth()
    
        
    # Create a new material with a specific color
    smpl_material_name = f"{uuid.uuid4()}"
    if smpl_material_name in bpy.data.materials:
        smpl_material = bpy.data.materials[smpl_material_name]
    else:
        smpl_material = bpy.data.materials.new(name=smpl_material_name)

    # Set the material color (RGBA)
    smpl_material.use_nodes = True
    bsdf_node = smpl_material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = get_smpl_frame_color(index + 1, len(obj_files))
#        bsdf_node.inputs['Metallic'].default_value = 0.3
        bsdf_node.inputs['Roughness'].default_value = 0.265
        
    
     # Apply the created material to the smpl
    if len(smpl.data.materials):
        # Replace existing materials if there are any
        smpl.data.materials[0] = smpl_material
    else:
        # Add the material if the object has no materials
        smpl.data.materials.append(smpl_material)

