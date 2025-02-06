import bpy
import os
import math
import matplotlib.colors
import numpy as np
import logging


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



obj_directory = base_dir + ours_1_0



output_directory = "C:/animations"
animation_file_name = obj_directory.split("/")[-2] + ".mp4"



color_background = "#E2E2E2"
color_smpl = "#0080D3"
color_plane1 = "#444444"
color_plane2 = "#121212"
color_plane3 = "#E2E2E2"


def hex_to_rgb(hex_color):
    rgb = list(matplotlib.colors.to_rgb(hex_color))
    rgb.append(1.0)
    return tuple(rgb)


def hsv_to_rgb(h, s, v):
    rgb = list(matplotlib.colors.hsv_to_rgb((h, s, v)))
    rgb.append(1.0)
    return tuple(rgb)





obj_files = sorted([f for f in os.listdir(obj_directory) if f.endswith(".obj")])


# Clear all existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Set up scene
scene = bpy.context.scene
frame_start = 1 


# Change the background color to gray
bpy.context.scene.world.use_nodes = True
world_nodes = bpy.context.scene.world.node_tree.nodes
background_node = world_nodes.get("Background")
if background_node:
    background_node.inputs[0].default_value = hex_to_rgb(color_background)
    




# Create a new material with a specific color
smpl_material_name = f"SMPLMaterial"
if smpl_material_name in bpy.data.materials:
    smpl_material = bpy.data.materials[smpl_material_name]
else:
    smpl_material = bpy.data.materials.new(name=smpl_material_name)

# Set the material color (RGBA)
smpl_material.use_nodes = True
smpl_bsdf_node = smpl_material.node_tree.nodes.get('Principled BSDF')
if smpl_bsdf_node:
    smpl_bsdf_node.inputs['Base Color'].default_value = hex_to_rgb(color_smpl)
    smpl_bsdf_node.inputs['Metallic'].default_value = 0.3
    smpl_bsdf_node.inputs['Roughness'].default_value = 0.7
    


        
        
  
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
    
    
plane3_material_name = "Plane3Material"
if plane3_material_name in bpy.data.materials:
    plane3_material = bpy.data.materials[plane3_material_name]
else:
    plane3_material = bpy.data.materials.new(name=plane3_material_name)
    
plane3_material.use_nodes = True
plane3_bsdf_node = plane3_material.node_tree.nodes.get('Principled BSDF')
if plane3_bsdf_node:
    plane3_bsdf_node.inputs['Base Color'].default_value = hex_to_rgb(color_plane3)  
    
    
    
    
    
    

# List to keep track of all imported objects
smpls = []

# Loop through all OBJ files and import them
for index, obj_file in enumerate(obj_files):
    # Import the OBJ file using wm.obj_import
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_directory, obj_file))

    # Get the newly imported object
    smpl = bpy.context.selected_objects[0]
    smpls.append(smpl)
    bpy.ops.object.shade_smooth()
    
    # Apply the created material to the smpl
    if len(smpl.data.materials):
        # Replace existing materials if there are any
        smpl.data.materials[0] = smpl_material
    else:
        # Add the material if the object has no materials
        smpl.data.materials.append(smpl_material)
    


# Set visibility keyframes
for index, smpl in enumerate(smpls):
    frame_number = frame_start + index

    # Set the visibility for the current frame
    for obj in smpls:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
        obj.keyframe_insert(data_path="hide_render", frame=frame_number)

    # Only make the current object visible
    smpl.hide_viewport = False
    smpl.hide_render = False
    smpl.keyframe_insert(data_path="hide_viewport", frame=frame_number)
    smpl.keyframe_insert(data_path="hide_render", frame=frame_number)

# Set the end frame of the animation to the total number of OBJ files
scene.frame_end = frame_start + len(obj_files) - 1








bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(-1.35, -0.35, -1.179), scale=(1, 1, 1))

bpy.ops.transform.resize(value=(14.0, 14.0, 14.0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

plane3 = bpy.context.selected_objects[0]
if len(plane3.data.materials):
    plane3.data.materials[0] = plane3_material
else:
    plane3.data.materials.append(plane3_material)



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
light_data = bpy.data.lights.new(name="Main Light", type='AREA')
light_data.energy = 10
light_object = bpy.data.objects.new(name="Main Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)


bpy.context.view_layer.objects.active = light_object
light_object.location = (-1.26867, -0.332561, 1.87827)
light_object.scale = (16,16,16)


# Add Sun Light
sun_light_data = bpy.data.lights.new(name="Sun Light", type='SUN')
sun_light_data.angle = 1.93
sun_light_data.energy = 1.5
sun_light_object = bpy.data.objects.new(name="Sun Light", object_data=sun_light_data)
bpy.context.collection.objects.link(sun_light_object)

bpy.context.view_layer.objects.active = sun_light_object
sun_light_object.location = (-1.7138, -2.8286, 4.12933)
sun_light_object.rotation_euler= (18.4873, 20.1977, -30.1056)














# Set up render settings to export animation as a video

scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
scene.render.filepath = os.path.join(output_directory, animation_file_name)
scene.render.fps = 24  # Set your desired frames per second
scene.frame_start = frame_start  # Set the start frame
scene.frame_end = frame_start + len(obj_files) - 1  # Set the end frame

print("Animation setup completed.")

# Render the animation to the output video file
#bpy.ops.render.render(animation=True)
