import bpy
import os
import math
import matplotlib.colors
import numpy as np
import logging


COUNT = 8
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

obj_directory = base_dir + results_4_1




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


#def get_alphas(count, min=70, max=100):
#    lst = [i + min for i in list(range(0, max-min + 1))]
#    if count <= 1:
#        return [lst[0], lst[-1]]
#    step = (len(lst) - 1) / (count - 1)
#    selected_numbers = [lst[0]]  # Include the first element
#    for i in range(1, count - 1):  # Select the intermediate elements
#        selected_numbers.append(lst[int(i * step)])
#    selected_numbers.append(lst[-1])  # Include the last element
#    return [i/100.0 for i in selected_numbers]



def get_alphas(count, min=70, max=100):
    if count <= 1:
        return [lst[0], lst[-1]]
    
    return [i/100.0 for i in np.linspace(min, max, count)]






obj_files = sorted([f for f in os.listdir(obj_directory) if f.endswith(".obj")])
num_frames = len(obj_files)

frame_indices = get_frame_indices(num_frames, COUNT) 
alphas = get_alphas(COUNT, min=smpl_opacity_min, max=smpl_opacity_max)

obj_files = [obj_files[i] for i in frame_indices]

# Clear all existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Set up scene
scene = bpy.context.scene



# Change the background color to gray
bpy.context.scene.world.use_nodes = True
world_nodes = bpy.context.scene.world.node_tree.nodes
background_node = world_nodes.get("Background")
if background_node:
    background_node.inputs[0].default_value = hex_to_rgb(color_background)

## Create a new material with a specific color
#material_name = "ImportedObjMaterial"
#if material_name in bpy.data.materials:
#    material = bpy.data.materials[material_name]
#else:
#    material = bpy.data.materials.new(name=material_name)

## Set the material color (RGBA)
#material.use_nodes = True
#bsdf_node = material.node_tree.nodes.get('Principled BSDF')
#if bsdf_node:
#    bsdf_node.inputs['Base Color'].default_value = hex_to_rgb(color_smpl)
#    bsdf_node.inputs['Metallic'].default_value = 0.3
#    bsdf_node.inputs['Roughness'].default_value = 0.7



# Create a new material with a specific color
plane1_material_name = "Plane1Material"
if plane1_material_name in bpy.data.materials:
    plane1_material = bpy.data.materials[plane1_material_name]
else:
    plane1_material = bpy.data.materials.new(name=plane1_material_name)
    
plane1_material.use_nodes = True
plane1_bsdf_node = plane1_material.node_tree.nodes.get('Principled BSDF')
if plane1_bsdf_node:
    plane1_bsdf_node.inputs['Base Color'].default_value = hex_to_rgb(color_plane1)  


plane2_material_name = "Plane2Material"
if plane2_material_name in bpy.data.materials:
    plane2_material = bpy.data.materials[plane2_material_name]
else:
    plane2_material = bpy.data.materials.new(name=plane2_material_name)
    
plane2_material.use_nodes = True
plane2_bsdf_node = plane2_material.node_tree.nodes.get('Principled BSDF')
if plane2_bsdf_node:
    plane2_bsdf_node.inputs['Base Color'].default_value = hex_to_rgb(color_plane2)  
    
    
    
    

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
    smpl_material_name = f"SMPLMaterial_{index}"
    if smpl_material_name in bpy.data.materials:
        smpl_material = bpy.data.materials[smpl_material_name]
    else:
        smpl_material = bpy.data.materials.new(name=smpl_material_name)

    # Set the material color (RGBA)
    smpl_material.use_nodes = True
    bsdf_node = smpl_material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = hsv_to_rgb(0.611, alphas[index], 0.65)
        bsdf_node.inputs['Metallic'].default_value = 0.3
        bsdf_node.inputs['Roughness'].default_value = 0.7
        
    
     # Apply the created material to the smpl
    if len(smpl.data.materials):
        # Replace existing materials if there are any
        smpl.data.materials[0] = smpl_material
    else:
        # Add the material if the object has no materials
        smpl.data.materials.append(smpl_material)







bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(-1.35, -0.35, -1.177), scale=(1, 1, 1))

bpy.ops.transform.resize(value=(2.68, 2.68, 2.68), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

plane1 = bpy.context.selected_objects[0]
if len(plane1.data.materials):
    plane1.data.materials[0] = plane1_material
else:
    plane1.data.materials.append(plane1_material)
    
 
    
bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(-1.35, -0.35, -1.172), scale=(1, 1, 1))

bpy.ops.transform.resize(value=(2.405, 2.405, 2.405), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

plane2 = bpy.context.selected_objects[0]
if len(plane2.data.materials):
    plane2.data.materials[0] = plane2_material
else:
    plane2.data.materials.append(plane2_material)
    
    
# Add Area Light 1
    
#bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=(-1.26867, -0.332561, 1.87827), scale=(3.523, 3.523, 3.523))
#bpy.ops.transform.resize(value=(3.523, 3.523, 3.523), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
#bpy.context.space_data.context = 'DATA'
#bpy.context.object.data.energy = 50



# Add Area Light 1
light_data = bpy.data.lights.new(name="Main Light", type='AREA')
light_data.energy = 100
light_object = bpy.data.objects.new(name="Main Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)


bpy.context.view_layer.objects.active = light_object
light_object.location = (-1.26867, -0.332561, 1.87827)
light_object.scale = (16,16,16)


# Add Sun Light
sun_light_data = bpy.data.lights.new(name="Sun Light", type='SUN')
sun_light_data.angle = 1.93
sun_light_data.energy = 5.0
sun_light_object = bpy.data.objects.new(name="Sun Light", object_data=sun_light_data)
bpy.context.collection.objects.link(sun_light_object)

bpy.context.view_layer.objects.active = sun_light_object
sun_light_object.location = (-1.7138, -2.8286, 4.12933)
sun_light_object.rotation_euler= (18.4873, 20.1977, -30.1056)


#def draw_sphere(location, radius=1, color=(1, 1, 1, 1)):

#    # Add a UV Sphere at the specified location
#    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=location)
#    
#    # Get the active object (the newly created sphere)
#    obj = bpy.context.active_object
#    
#    obj.scale = (radius,radius,radius)
#    obj.location = location
#    
#    # Create a new material and set color
#    material = bpy.data.materials.new(name="SphereMaterial")
#    material.use_nodes = True
#    bsdf = material.node_tree.nodes["Principled BSDF"]
#    bsdf.inputs["Base Color"].default_value = color  # Set the color (RGBA)
#    
#    # Assign material to the object
#    if obj.data.materials:
#        obj.data.materials[0] = material
#    else:
#        obj.data.materials.append(material)

## Example Usage:
#location = (0, 0, 0)  # Center of the sphere
#color = (1, 1, 1, 1)  # Green color (RGBA)


#for smpl in smpls:
#    draw_sphere((smpl.matrix_world.to_translation()[0], -0.24,-1.12), color=color)
