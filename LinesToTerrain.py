from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import call

import numpy as np
import imageio
imageio.plugins.freeimage.download()

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pyrr import Matrix44

import json
import base64

import moderngl
import moderngl_window as mglw
from moderngl_window.scene.camera import KeyboardCamera, OrbitCamera

from PIL import Image, ImageTk, ImageDraw, ImageGrab

from tkinter import Tk, Button, Canvas, Scale, OptionMenu, StringVar, PhotoImage, HORIZONTAL, NW, ROUND, TRUE, RAISED, SUNKEN
#from tkinter import *
from tkinter.ttk import Progressbar, Style

from pathlib import Path

from threading import Thread, Event
from time import sleep


def terrain(size):
    vertices = np.dstack(np.mgrid[0:size, 0:size][::-1]) / size
    temp = np.dstack([np.arange(0, size * size - size), np.arange(size, size * size)])
    index = np.pad(temp.reshape(size - 1, 2 * size), [[0, 0], [0, 1]], 'constant', constant_values=-1)
    return vertices, index

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Multithread events
closeEvent = Event()
MLEvent = Event()
refreshEvent = Event()
resetEvent = Event()

# Main Drawing UI Here
class Paint(object):

    DEFAULT_BRUSH_SIZE = 5.0
    DEFAULT_COLOR = 'red'

    def __init__(self):
        self.root = Tk()

        self.root.winfo_toplevel().title("Lines2Terrain")

        #Toolbar Buttons
        self.peak_button = Button(self.root, text='▲', foreground="red", width=2, height=1, command=self.use_peak)
        self.peak_button.grid(row=0, column=5)

        self.valley_button = Button(self.root, text='▼', foreground="blue", width=2, height=1, command=self.use_valley)
        self.valley_button.grid(row=0, column=6)

        self.eraser_button = Button(self.root, text='⌫', foreground="#e3a6d1", width=2, height=1, command=self.use_eraser)
        self.eraser_button.grid(row=0, column=7)

        self.clear_button = Button(self.root, text='❌', foreground="red", width=2, height=1, command=self.clear)
        self.clear_button.grid(row=0, column=8)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL, sliderlength=25, width= 13, bd=0, troughcolor="#111", resolution=0.25, font=("DejaVu Sans", 8))
        self.choose_size_button.grid(row=0, column=18)

        self.progress = Progressbar(self.root, orient = HORIZONTAL, length = 28, mode = 'determinate', )
        self.progress.grid(row=0, column=1520)
        #Model Selection dropdown
        self.selectedModel = StringVar(self.root)
        modelOptions = []

        for (dirpath, dirnames, filenames) in os.walk("Models"):
            modelOptions.extend(dirnames)
            break

        self.selectedModel.set(modelOptions[0]) # set the default option
        global model
        model = self.selectedModel.get()
        
        self.modelMenu = OptionMenu(self.root, self.selectedModel, *modelOptions)
        self.modelMenu.grid(row=0, column=1530)

        # link function to change dropdown
        self.selectedModel.trace('w', self.change_dropdown)

        #canvas 1 - Drawing
        self.c = Canvas(self.root, bg='black', width=512, height=512)
        self.c.grid(row=1, column=0, columnspan=512)

        #canvas 2 - ImagePreview
        self.c2 = Canvas(self.root, bg='gray', width=512, height=512)
        self.c2.grid(row=1, column=512, columnspan=1024)
        
        #location to start the window and window size
        self.root.geometry('1032x551+200+50')

        #Methods
        self.setup()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.choose_size_button.set(2)
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.peak_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.update)
        self.c.bind('<ButtonRelease-3>', self.updateRight)

    
    def updateC2(self):
        if MLEvent.is_set():
            self.root.after(1000, self.updateC2)
        else:
            self.image = PhotoImage(file='data/out.png')
            self.c2.create_image(2, 2, image=self.image, anchor=NW)
            self.progress.stop()
            refreshEvent.set()

    def use_peak(self):
        self.activate_button(self.peak_button)
        self.color = 'red'

    def use_valley(self):
        self.activate_button(self.valley_button)
        self.color = 'blue'

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def clear(self):
        self.c.delete('all')
        self.c2.delete('all')
        self.image = PhotoImage(file='data/blank.png')
        self.c2.create_image(2, 2, image=self.image, anchor=NW)
        resetEvent.set()

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'black' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
    
    # on change dropdown value
    def change_dropdown(self, *args):
        global model
        model = self.selectedModel.get()
        print( "Running " + self.selectedModel.get() + " Model:" + model)
        self.progress.start(interval=50)
        MLEvent.set()
        # start update preview image loop
        self.root.after_idle(self.updateC2)


    def update(self, event):
        self.old_x, self.old_y = None, None

        self.progress.start(interval=50)

        #save drawn image
        x=self.root.winfo_rootx()+self.c.winfo_x()
        y=self.root.winfo_rooty()+self.c.winfo_y()
        x1=x+self.c.winfo_width()
        y1=y+self.c.winfo_height()
        ImageGrab.grab().crop((x+2,y+2,x1-2,y1-2)).save("data/in.png")
        
        #Run ML model
        MLEvent.set()
        
        # start update preview image loop
        self.root.after_idle(self.updateC2)      

    def updateRight(self, event):
        self.c.delete('all')

    def on_closing(self):
        closeEvent.set()
        self.root.destroy()
        
# Run ML Model Class        
class MLModel(object):
    
    def __init__(self):

        self.runML()

    def processModel(self, input_file, output_file, model_dir):
        with open(input_file, "rb") as f:
            input_data = f.read()

        input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
        input_instance = json.loads(json.dumps(input_instance))

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_dir + "/export.meta")
            saver.restore(sess, model_dir + "/export")
            input_vars = json.loads(tf.get_collection("inputs")[0])
            output_vars = json.loads(tf.get_collection("outputs")[0])
            input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
            output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

            input_value = np.array(input_instance["input"])
            output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]
        
        output_instance = dict(output=output_value.decode("ascii"), key="0")

        data = output_instance["output"]

        b64data = output_instance["output"]
        b64data += "=" * (-len(b64data) % 4)
        output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

        with open(output_file, "wb") as f:
            f.write(output_data)
    
    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def scale_to_range(self, x, min_val, max_val, a=-1, b=1):
        return ((b - a) * (x - min_val)) / (max_val - min_val) + a

    def runML(self):
        while True:
            if closeEvent.is_set():
                break
            if MLEvent.is_set():
                self.processModel("data/in.png", "data/out.png", "Models/"+model)
                im = imageio.imread("data/out.png", format='PNG-FI')
                grayscale = self.rgb2gray(im)
                #level = self.scale_to_range(grayscale, 5140, 60395, 0, 2**16-1)
                imageio.imwrite("data/out2.png", grayscale.astype(np.uint16))
                MLEvent.clear()
            sleep(1)

# 3D Preview
class WireframeTerrain(mglw.WindowConfig):
    title = " "
    gl_version = (3, 3)
    window_size = (1032, 580)
    aspect_ratio = 16 / 9
    samples = 4
    resizable = True
    resource_dir = (Path(__file__) / '../data').absolute()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.hasrendered = False
        self.wnd.position = (208, 664)

        self.camera = OrbitCamera(target=(0.,0.,0.15), radius=(1.2), angles=(-90.,-145.), aspect_ratio=self.wnd.aspect_ratio)

        self.rotation_enabled = True
        self.wnd.mouse_exclusivity = False

        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 5
        self.camera._angle_y = 0
    
        self.texture = self.load_texture_2d("blank.png")
        self.imgmax = 0.501
        self.imgmin = 0.499

        self.colorramp = self.load_texture_2d('clt.png')
        
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;
                uniform sampler2D Heightmap;

                in vec2 in_vert;
                in vec3 in_color;

                out float map_height;

                void main() {
                    float height = texture(Heightmap, in_vert.xy).r - 0.3;
                    map_height = height + 0.3;
                    gl_Position = Mvp * vec4(in_vert.xy - 0.5, height, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                in float map_height;
                out vec4 f_color;
                uniform sampler2D colorramp;
                uniform bool use_colorramp;
                uniform vec2 range;

                void main() {
                    if (use_colorramp){
                        f_color = texture(colorramp, vec2(0.5, (map_height - range[0]) / (range[1] - range[0]) ));
                        
                    }
                    else {
                        f_color = vec4(vec3(0.), 1.);
                    }
                }
            ''',
        )

        self.prog['colorramp'] = 1
        self._using_colorramp = False
        self.prog['use_colorramp'] = self._using_colorramp

        self.prog['range'] = self.imgmin, self.imgmax

        self.mvp = self.prog['Mvp']

        vertices, index = terrain(128)

        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(index.astype('i4').tobytes())

        vao_content = [
            (self.vbo, '2f', 'in_vert'),
        ]

        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)

    #mouse events
    def mouse_press_event(self, x: int, y: int, button: int):
        if self.wnd.mouse_states.right:
            self.rotation_enabled = not self.rotation_enabled

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self._using_colorramp = not self._using_colorramp
                self.prog['use_colorramp'] = self._using_colorramp

    def mouse_drag_event(self, x: int, y: int, dx, dy):
        
        if self.wnd.mouse_states.left:
            self.camera._angle_x += dx * self.camera.mouse_sensitivity / 100.
            self.camera._angle_y += dy * self.camera.mouse_sensitivity / 500.

            # clamp the y angle to avoid weird rotations
            self.camera._angle_y = max(min(self.camera.angle_y, 0.1), -0.4)
            #self.camera.rot_state(dx, dy)

    #render call
    def render(self, time, frame_time):
        
        if self.rotation_enabled:
            self.camera._angle_x = self.camera.angle_x - .001

        self.ctx.clear(0.7, 0.7, 0.7)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = not self._using_colorramp
        self.prog['range'] = self.imgmin, self.imgmax

        projection_matrix = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        
        camera_matrix = Matrix44.look_at(
            (np.cos(-self.camera.angle_x), np.sin(-self.camera.angle_x), 0.8 -self.camera.angle_y),
            (0.0, 0.0, 0.1),
            (0.0, 0.0, 1.0),
        )
        
        #to update texture every frame
        if refreshEvent.is_set():
            print("load texture")
            self.texture = self.load_texture_2d("out.png")
            self.imgmin = (np.amin(np.array(imageio.imread('data/out.png')))/255.0)
            print(self.imgmin)
            self.imgmax = (np.amax(np.array(imageio.imread('data/out.png')))/255.0)
            print(self.imgmax)
            refreshEvent.clear()

        #clear preview
        if resetEvent.is_set():
            print("clear 3d")
            self.texture = self.load_texture_2d("blank.png")
            self.imgmax = 0.501
            self.imgmin = 0.499
            resetEvent.clear()

        self.colorramp.use(1)
        self.texture.use(0)
        
        #draw
        self.mvp.write((projection_matrix * camera_matrix).astype('f4').tobytes())
        
        #set render mode
        self.vao.render(moderngl.TRIANGLE_STRIP)

        #close if Paint thread is closed
        if closeEvent.is_set():
            self.wnd.close()
       

if __name__ == '__main__':

    t1 = Thread(target = Paint)
    t1.start()
    
    t2 = Thread(target = MLModel, daemon=True)
    t2.daemon = True
    t2.start()

    t3 = Thread(target = WireframeTerrain.run(), daemon=True)
    #t3.daemon = True
    t3.start()
    
    t1.join()
    print("closed T1")
    t2.join()
    print("closed T2")
    t3.join()
    print("closed T3")