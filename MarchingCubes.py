import sys
import numpy as np
import math
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import lookup
import pyrr.matrix44 as m44


class RenderMarchingCube : 

	DISPLAYMODE_MESH = 1
	
	def __init__(self, screenWidth=800, screenHeight=600, worldSize = 10, radius=20):
		self.FPS = 30	
		self.displayMode = None
		self.dragging = False


		# Set world
		self.worldSize = int(worldSize+1)
	
		#perform marching cubes
		self.vertices = []
		self.stepSize = 0.9
		self.isolevel= 0
		minVal = -4
		maxVal = self.worldSize-3
		self.findPolygons(minVal, maxVal, self.stepSize, self.isolevel)

		# Compute the normals
		self.normals = self.compute_normals(self.vertices)

		self.writePLY(self.vertices, self.normals, "testfile1")
		
		# create openGL window
		pygame.init()
		pygame.display.set_caption('Marching Cubes Visualizer')
		self.displaySize = (screenWidth, screenHeight)
		pygame.display.set_mode(self.displaySize, DOUBLEBUF | OPENGL)
		self.clock = pygame.time.Clock()
		
		# configure openGL perspective
		glMatrixMode(GL_PROJECTION)
		gluPerspective(45, (screenWidth/screenHeight), 0.1, 50.0)
		
		# configure model view
		glMatrixMode(GL_MODELVIEW)
		glEnable(GL_CULL_FACE)		#enable culling
		glCullFace(GL_BACK)
		glEnable(GL_DEPTH_TEST)		#enable depth test

		# initial camera setup 
		self.polarCamera = [radius, 0, 90]	#(r, theta, phi)
		self.polarCameraToCartesian()
		self.cameraTarget = [0, 0, 0]
		self.up = [0, 1, 0]
		glLoadIdentity()
		gluLookAt(*self.cameraPosition, *self.cameraTarget, *self.up)
		


	def mainLoop(self):
		"""
		Main loop of Marching Cubes, handles camera movement through keyboard and mouse
		Creates and compiles the shaders
		Initialize and set the VAO, VBO to draw triangles.
		"""

		while True:
			# cap FPS at 30
			self.dt = self.clock.tick(self.FPS)

			# pygame event loop for quit and mouse scrollwheel
			for event in pygame.event.get():
				#handle quit event
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
				# handle mouse scrollwheel (adjust threshold)
				if event.type == pygame.MOUSEBUTTONDOWN:
					if event.button == 1:
						self.mouse_drag_start = pygame.mouse.get_pos()
						self.dragging = True
				elif event.type == pygame.MOUSEBUTTONUP:
					if event.button == 1:  # Left mouse button
						self.dragging = False
				elif event.type == pygame.MOUSEMOTION and self.dragging:
					dx = event.pos[0] - self.mouse_drag_start[0]
					self.mouse_drag_start = event.pos
					self.polarCamera[1] += 0.08 * dx * self.dt
					if self.polarCamera[1] > 180:
						self.polarCamera[1] -= 360
					elif self.polarCamera[1] <= -180:
						self.polarCamera[1] += 360

					
			# handle camera controller
			self.keyboardController()

			vertex_shader_source = '''
			#version 330 core
			layout (location = 0) in vec3 aPos;
			layout (location = 1) in vec3 aNormal;

			out vec3 FragPos;
			out vec3 Normal;

			uniform mat4 MVP;
			uniform mat4 V;

			void main()
			{
				gl_Position = MVP * vec4(aPos, 1.0);
				FragPos = (V * vec4(aPos, 1.0)).xyz;
				Normal = mat3(transpose(inverse(V))) * aNormal;
			}
			'''

			fragment_shader_source = '''
			#version 330 core
			in vec3 FragPos;
			in vec3 Normal;

			out vec4 FragColor;

			uniform vec3 LightDir;
			uniform vec3 modelColor;

			const vec3 ambientColor = vec3(0.2, 0.2, 0.2);
			const vec3 specularColor = vec3(1, 1, 1);
			const float shininess = 64.0;

			void main()
			{
				// Ambient
				vec3 ambient = ambientColor * modelColor;

				// Diffuse
				vec3 norm = normalize(Normal);
				vec3 lightDir = normalize(-LightDir);
				float diff = max(dot(norm, lightDir), 0.0);
				vec3 diffuse = diff * modelColor;

				// Specular
				vec3 viewDir = normalize(-FragPos);
				vec3 reflectDir = reflect(-lightDir, norm);
				float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
				vec3 specular = specularColor * spec;

				// Final color
				vec3 finalColor = ambient + diffuse + specular;
				FragColor = vec4(finalColor, 1.0);
			}
			'''


			shader_program = self.compile_shaders(vertex_shader_source, fragment_shader_source)

			# Init VAO and VBOs
			vao, num_triangles = self.initialize_vao(self.vertices, self.normals)
			model_matrix = np.identity(4, dtype=np.float32)
			
			# Set up view, projection and light matrix
			gluLookAt(*self.cameraPosition, *self.cameraTarget, *self.up)
			fov = 45
			aspect_ratio = 800.0 / 600.0
			near_plane = 0.1
			far_plane = 100
			projection_matrix = m44.create_perspective_projection(fov, aspect_ratio, near_plane, far_plane)
			
			light_direction = np.array([0.0, -1.0, 1.0], dtype=np.float32)  # Set up your light direction
			light_direction = light_direction / np.linalg.norm(light_direction)

			view_matrix = np.zeros((4, 4), dtype=np.float32)
			glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix)
			
			# Set the MVP, V, and LightDir uniform variables
			MVP = np.dot(projection_matrix, np.dot(view_matrix, model_matrix))

			# glUniformMatrix4fv(glGetUniformLocation(shader_program, "MVP"), 1, GL_FALSE, MVP)
			# glUniformMatrix4fv(glGetUniformLocation(shader_program, "V"), 1, GL_FALSE, view_matrix)
			# glUniform3fv(glGetUniformLocation(shader_program, "LightDir"), 1, light_direction)

			# Set the modelColor uniform variable

			# model_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
			# glUniform3fv(glGetUniformLocation(shader_program, "modelColor"), 1, model_color)
			
			#Draw scene
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			self.renderScene()
			self.draw_triangles(vao, num_triangles, shader_program, model_matrix, view_matrix, projection_matrix, light_direction)
			
			#look at and render scene
			glLoadIdentity()

			pygame.display.flip()


	def polarCameraToCartesian(self):
		"""
		Converts from polar position to cartesian
		"""
		x = self.polarCamera[0]*np.sin(self.polarCamera[1]*np.pi/180)*np.sin(self.polarCamera[2]*np.pi/180)
		y = self.polarCamera[0]*np.cos(self.polarCamera[2]*np.pi/180)
		z = self.polarCamera[0]*np.cos(self.polarCamera[1]*np.pi/180)*np.sin(self.polarCamera[2]*np.pi/180)
		self.cameraPosition = [x, y, z]


	def keyboardController(self):
		"""
		Processes inputs
		"""
		# get pressed keys
		keypress = pygame.key.get_pressed()

		# Render the Mesh
		if keypress[pygame.K_o]:
			self.displayMode = self.DISPLAYMODE_MESH
			glDisable(GL_CULL_FACE)	#disable culling for mesh viewing

		#apply polar camera movement
		if keypress[pygame.K_UP]:
			self.polarCamera[0] -= 0.01 * self.dt
			if self.polarCamera[0] < 1:
				self.polarCamera[0] = 1.0
		if keypress[pygame.K_DOWN]:
			self.polarCamera[0] += 0.01 * self.dt
			if self.polarCamera[0] > 179:
				self.polarCamera[0] = 179
		if keypress[pygame.K_w]:
			self.polarCamera[2] -= 0.08 * self.dt
			if self.polarCamera[2] < 1:
				self.polarCamera[2] = 1.0
		if keypress[pygame.K_s]:
			self.polarCamera[2] += 0.08 * self.dt
			if self.polarCamera[2] > 179:
				self.polarCamera[2] = 179
		if keypress[pygame.K_d]:
			self.polarCamera[1] += 0.08 * self.dt
			if self.polarCamera[1] > 180:
				self.polarCamera[1] -= 360
		if keypress[pygame.K_a]:
			self.polarCamera[1] -= 0.08 * self.dt
			if self.polarCamera[1] <= -180:
				self.polarCamera[1] += 360
		# update camera cartesian position
		self.polarCameraToCartesian()


	def renderScene(self):
		"""
		Renders scene:  Draws the box and the mesh
		"""
		glBegin(GL_LINES)
		# draw axes
		glColor3f(1, 0, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(self.worldSize/2, 0, 0)
		glColor3f(0, 1, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(0, self.worldSize/2, 0)
		glColor3f(0, 0, 1)
		glVertex3f(0, 0, 0)
		glVertex3f(0, 0, self.worldSize/2)
		# draw bounding box
		glColor3f(1,1,1)
		scalar = (self.worldSize-1)/2
		for x in [-1, 1]:
			for y in [-1,1]:
				for z in [-1,1]:
					glVertex3f(scalar*x, scalar*y, scalar*z)
		for z in [-1, 1]:
			for x in [-1,1]:
				for y in [-1,1]:
					glVertex3f(scalar*x, scalar*y, scalar*z)
		for y in [-1, 1]:
			for z in [-1,1]:
				for x in [-1,1]:
					glVertex3f(scalar*x, scalar*y, scalar*z)
		glEnd()

		# draw mesh  
		if self.displayMode is self.DISPLAYMODE_MESH:
			offset = int(self.worldSize/2)
			glBegin(GL_TRIANGLES)
			glColor3f(0.64,2.24,2.08)
			for vertex in self.vertices:
				glVertex3f(offset + vertex[0][0] - 3, offset + vertex[0][1] - 5, offset + vertex[0][2] -  2)
				glVertex3f(offset + vertex[1][0] - 3, offset + vertex[1][1] - 5, offset + vertex[1][2]  - 2)
				glVertex3f(offset + vertex[2][0] - 3, offset + vertex[2][1] - 5, offset + vertex[2][2]  - 2)

			glEnd()

		# draw background in the distance
		glLoadIdentity()
		glBegin(GL_QUADS)
		glColor3f(0,0,0)
	
		glVertex3f(-30, -23, -49.5)
		glVertex3f(30, -23, -49.5)
		glColor3f(184/256, 201/256, 242/256)
		glVertex3f(30, 23, -49.5)
		glVertex3f(-30, 23, -49.5)
		glEnd()


	def findPolygons(self, min_val, max_val, stepsize, isolevel):
		"""
		Perform MarchingCubesPolygons algorithm across the worldspace to generate an array of polygons to plot
		"""
		x = min_val
		y = min_val
		z = min_val

		while z < max_val:
			y = min_val
			x = min_val
			while y< max_val:
				x = min_val
				while x < max_val:
					p = [(min_val + x * stepsize, min_val + y * stepsize, min_val + z * stepsize),
						(min_val + (x + 1) * stepsize, min_val + y * stepsize, min_val + z * stepsize),
						(min_val + (x + 1) * stepsize, min_val + (y + 1) * stepsize, min_val + z * stepsize),
						(min_val + x * stepsize, min_val + (y + 1) * stepsize, min_val + z * stepsize),
						(min_val + x * stepsize, min_val + y * stepsize, min_val + (z + 1) * stepsize),
						(min_val + (x + 1) * stepsize, min_val + y * stepsize, min_val + (z + 1) * stepsize),
						(min_val + (x + 1) * stepsize, min_val + (y + 1) * stepsize, min_val + (z + 1) * stepsize),
						(min_val + x * stepsize, min_val + (y + 1) * stepsize, min_val + (z + 1) * stepsize)]
					self.marching_cubes(p, self.f,isolevel)
					x += stepsize
				y += stepsize
			z +=stepsize


	# Function 1 as described in assignment
	def f(self,x,y,z):
		return y-math.sin(x) * math.cos(z)
	
	# Function 2 as described in assignment
	def f2(self,x,y,z):
		return x**2 - y**2 - z**2 - z
	
	# Helperfunction to do vertex interpolation
	def VertexInterp(self,isolevel, p1, p2, valp1, valp2):
		if (abs(isolevel - valp1) < 0.00001):
			return p1
		if (abs(isolevel - valp2) < 0.00001):
			return p2
		if(abs(valp1 - valp2) < 0.00001):
			return p1
		mu = (isolevel - valp1) / (valp2 - valp1)
		x = p1[0] + mu * (p2[0] - p1[0])
		y = p1[1] + mu * (p2[1] - p1[1])
		z = p1[2] + mu * (p2[2] - p1[2])
		return (x,y,z)
	
	# Performs marching cube algorithm
	def marching_cubes(self,p, f, isolevel):
		cornervalues = [f(*p[0]), f(*p[1]), f(*p[2]), f(*p[3]),
						f(*p[4]), f(*p[5]), f(*p[6]), f(*p[7])]
					
		# determine which verticies are inside the isovalue
		cubeindex = 0
		if cornervalues[0] < isolevel: cubeindex = cubeindex | 1
		if cornervalues[1] < isolevel: cubeindex = cubeindex | 2
		if cornervalues[2] < isolevel: cubeindex = cubeindex | 4
		if cornervalues[3] < isolevel: cubeindex = cubeindex | 8
		if cornervalues[4] < isolevel: cubeindex = cubeindex | 16
		if cornervalues[5] < isolevel: cubeindex = cubeindex | 32
		if cornervalues[6] < isolevel: cubeindex = cubeindex | 64
		if cornervalues[7] < isolevel: cubeindex = cubeindex | 128
					
		# Cube is entirely in/out of the surface
		if lookup.EDGE_TABLE[cubeindex] == 0: return []
		vertlist = [[]]*12
		
		# Determine verticies where surface intersects the cube
		if(lookup.EDGE_TABLE[cubeindex] & 1):
			vertlist[0] = self.VertexInterp(isolevel, p[0], p[1], cornervalues[0], cornervalues[1])
		if(lookup.EDGE_TABLE[cubeindex] & 2):
			vertlist[1] = self.VertexInterp(isolevel, p[1], p[2], cornervalues[1], cornervalues[2])
		if(lookup.EDGE_TABLE[cubeindex] & 4):
			vertlist[2] = self.VertexInterp(isolevel, p[2], p[3], cornervalues[2], cornervalues[3])
		if(lookup.EDGE_TABLE[cubeindex] & 8):
			vertlist[3] = self.VertexInterp(isolevel, p[3], p[0], cornervalues[3], cornervalues[0])
		if(lookup.EDGE_TABLE[cubeindex] & 16):
			vertlist[4] = self.VertexInterp(isolevel, p[4], p[5], cornervalues[4], cornervalues[5])
		if(lookup.EDGE_TABLE[cubeindex] & 32):
			vertlist[5] = self.VertexInterp(isolevel, p[5], p[6], cornervalues[5], cornervalues[6])
		if(lookup.EDGE_TABLE[cubeindex] & 64):
			vertlist[6] = self.VertexInterp(isolevel, p[6], p[7], cornervalues[6], cornervalues[7])
		if(lookup.EDGE_TABLE[cubeindex] & 128):
			vertlist[7] = self.VertexInterp(isolevel, p[7], p[4], cornervalues[7], cornervalues[4])
		if(lookup.EDGE_TABLE[cubeindex] & 256):
			vertlist[8] = self.VertexInterp(isolevel, p[0], p[4], cornervalues[0], cornervalues[4])
		if(lookup.EDGE_TABLE[cubeindex] & 512):
			vertlist[9] = self.VertexInterp(isolevel, p[1], p[5], cornervalues[1], cornervalues[5])
		if(lookup.EDGE_TABLE[cubeindex] & 1024):
			vertlist[10] = self.VertexInterp(isolevel, p[2], p[6], cornervalues[2], cornervalues[6])
		if(lookup.EDGE_TABLE[cubeindex] & 2048):
			vertlist[11] = self.VertexInterp(isolevel, p[3], p[7], cornervalues[3], cornervalues[7])
					
		## Create triangles
		i = 0
		while lookup.TRI_TABLE[cubeindex][i] != -1:
			self.vertices.append([
				vertlist[lookup.TRI_TABLE[cubeindex][i]],
				vertlist[lookup.TRI_TABLE[cubeindex][i+1]],
				vertlist[lookup.TRI_TABLE[cubeindex][i+2]]		
							])
			i += 3


	# Helper function to normalize.
	def normalize(self,v):
		norm = np.linalg.norm(v)
		if norm == 0:
			return v
		return v / norm

	# Computes the normals of the vertices
	def compute_normals(self, triangles):
		normals = []		
		for triangle in triangles:
			v1, v2, v3 = triangle
			v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
			edge1 = v2 - v1
			edge2 = v3 - v1
	
			# Calculate the normal vector using cross product, assuming counter clockwise winding order
			normal = self.normalize(np.cross(edge1, edge2))

			# Repeat the normal vector for each vertex in the triangle
			normals.extend([normal] * 3)

		return normals
	
	# Writes the PLY file.
	def writePLY(self, vertices, normals, fileName):
		num_vertices = len(vertices) * 3

		with open(fileName, 'w') as f:
			f.write("ply\n")
			f.write("format ascii 1.0\n")
			f.write(f"element vertex {num_vertices}\n")
			f.write("property float x\n")
			f.write("property float y\n")
			f.write("property float z\n")
			f.write("property float nx\n")
			f.write("property float ny\n")
			f.write("property float nz\n")
			f.write("element face {}\n".format(len(vertices)))
			f.write("property list uchar int vertex_indices\n")
			f.write("end_header\n")

			for triangle, normal in zip(vertices, normals):
				for vertex in triangle:
					f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {normal[0]} {normal[1]} {normal[2]}\n")

			for i in range(0, num_vertices, 3):
				f.write(f"3 {i} {i+1} {i+2}\n")


	# Helper function to compile the shaders
	def compile_shaders(self,vertex_shader_source, fragment_shader_source):
		vertex_shader = glCreateShader(GL_VERTEX_SHADER)
		glShaderSource(vertex_shader, vertex_shader_source)
		glCompileShader(vertex_shader)

		fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
		glShaderSource(fragment_shader, fragment_shader_source)
		glCompileShader(fragment_shader)

		shader_program = glCreateProgram()
		glAttachShader(shader_program, vertex_shader)
		glAttachShader(shader_program, fragment_shader)
		glLinkProgram(shader_program)

		return shader_program
	
	# Helper function to init the VAO with the verticies and normals.
	def initialize_vao(self,vertices, normals):
		vertex_data = []
		for triangle, normal in zip(vertices, normals):
			for vertex in triangle:
				vertex_data.extend(vertex)
				vertex_data.extend(normal)

		vao = glGenVertexArrays(1)
		glBindVertexArray(vao)

		vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo)
		glBufferData(GL_ARRAY_BUFFER, np.array(vertex_data, dtype=np.float32), GL_STATIC_DRAW)

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
		glEnableVertexAttribArray(0)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
		glEnableVertexAttribArray(1)

		return vao, len(vertices) * 3
	
	# Helper function which helps render the triangle mesh with phong like shader
	def draw_triangles(self,vao, num_triangles, shader_program, model_matrix, view_matrix, projection_matrix, light_direction):
		glUseProgram(shader_program)

		glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model_matrix)
		glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, view_matrix)
		glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, projection_matrix)
		glUniform3fv(glGetUniformLocation(shader_program,"lightDirection"), 1, light_direction)
		glBindVertexArray(vao)
		glDrawArrays(GL_TRIANGLES, 0, num_triangles)
		glBindVertexArray(0)
		glUseProgram(0)

def main():
	# default parameters
	screenWidth=800 
	screenHeight=600
	worldSize = 10
	radius=20
	# start app
	App = RenderMarchingCube(screenWidth, screenHeight, worldSize, radius)
	App.mainLoop()


if __name__ == "__main__":
    main()
