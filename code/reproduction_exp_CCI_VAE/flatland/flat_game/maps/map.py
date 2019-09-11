import random as rand
from pygame import Rect
import numpy as np

texture_params = {
    'corridors_texture': {
        'type': 'color',
        'c': (0, 150, 50),
    }
}

from pygame.locals import *

import pygame
import sys

import shapely.geometry as geo
from utils.texture import *

CORRIDOR_SIZE = 25
MAX_ROOM_SIZE = 300
MIN_ROOM_SIZE = 120
WALL_SIZE = 10


def get_random_texture():
    # sets the color randomly
    #TODO add more options for the textures
    texture = {
        'type': 'color',
        'c': (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)),
    }
    return texture


#additional utils for shapely package


def draw_linearring(r, screen):
    array_of_points = np.array(r)
    for ii in range(1, array_of_points.shape[0]):
        pygame.draw.line(screen, (255, 255, 0), array_of_points[ii - 1, :], array_of_points[ii, :], 2)
    pygame.draw.line(screen, (255, 255, 0), array_of_points[-1, :], array_of_points[0, :], 2)


def draw_multilinestring(m, screen):
    try:
        sh = geo.shape(m)
    except ValueError:
        draw_linearring(m, screen)
        return
    try:
        for line in sh:
            draw_linearring(line, screen)
    except TypeError:
        draw_linearring(m, screen)


def convert_to_list_of_poly(list):
    newlist = []
    for ll in list:
        if type(ll[0]) is geo.MultiPolygon:
            for poly in ll[0].geoms:
                newlist.append(Wall(poly, ll[1]))
        else:
            newlist.append(Wall(ll[0], ll[1]))
    return newlist


class Wall:
    def __init__(self, poly, texture = texture_params['corridors_texture']):
        [self.x, self.y] = np.array(poly.centroid)

        seg = []
        co = poly.minimum_rotated_rectangle.exterior.coords
        for ii in range(1, len(co)):
            seg.append(geo.LineString([co[ii - 1], co[ii]]))

        self.width = seg[0].distance(seg[2])
        self.height = seg[1].distance(seg[3])

        self.texture = texture


class Rect_line(Rect):
    def __init__(self, left, top, w, h):
        Rect.__init__(self, left, top, w, h)
        self.ring = geo.Polygon([(left, top), (left + w, top), (left + w, top + h), (left, top + h)])

        self.wall_polygons =[]

    def inflate_ip(self, x, y):
        Rect.inflate_ip(self, x, y)
        self.ring = geo.Polygon([(self.left, self.top), (self.left + self.w, self.top), (self.left + self.w, self.top + self.h), (self.left, self.top + self.h)])

    def create_walls(self):
        #function to define outer boundaries of the room
        inner_points = self.ring.exterior.coords
        inner_segments = []

        outer_line = Rect_line(self.left, self.top, self.w, self.h)
        outer_line.inflate_ip(WALL_SIZE*2, WALL_SIZE*2)

        outer_points = outer_line.ring.exterior.coords
        outer_segments = []

        for ii in range(1, len(inner_points)):
             inner_segments.append(geo.LineString([inner_points[ii - 1], inner_points[ii]]))

        for ii in range(1, len(outer_points)):
            outer_segments.append(geo.LineString([outer_points[ii - 1], outer_points[ii]]))

        for o_ii in range(len(outer_segments)):
            for i_ii in range(len(inner_segments)):
                dist1 = outer_segments[o_ii].distance(geo.Point(inner_segments[i_ii].coords[0]))
                dist2 = outer_segments[o_ii].distance(geo.Point(inner_segments[i_ii].coords[1]))
                if dist1 == WALL_SIZE and dist2 == WALL_SIZE:
                     poly = geo.MultiLineString([outer_segments[o_ii], inner_segments[i_ii]])
                     self.wall_polygons.append(poly.minimum_rotated_rectangle)


class Room(Rect_line):
    def __init__(self, centroid, w, h):
        left = centroid[0] - w/2
        top = centroid[1] - h/2
        Rect_line.__init__(self, left, top, w, h)

        self.corridors = []

        self.centroid = self.ring.centroid

        self.texture = get_random_texture()

        self.create_walls()

    def get_center(self):
        return self.left + self.width / 2, self.top + self.height / 2


class Corridor(object):
    narrow = CORRIDOR_SIZE
    texture = texture_params['corridors_texture']

    def __init__(self, room1, room2, mode = 'vh'):
        #vh - vertical first, horizontal second
        #hv - horizontal first, vertical second
        self.rooms = [room1, room2]

        if mode == 'vh':
            self.v = self.create_v(np.array(room1.centroid), np.array(room2.centroid))
            self.room_v = room1
            self.h = self.create_h(np.array(room2.centroid), np.array(room1.centroid))
            self.room_h = room2
        else:
            self.v = self.create_h(np.array(room1.centroid), np.array(room2.centroid))
            self.room_h = room1
            self.h = self.create_v(np.array(room2.centroid), np.array(room1.centroid))
            self.room_v = room2

        ring = self.h.ring.union(self.v.ring)
        ring = ring.difference(room1.ring)
        self.ring = ring.difference(room2.ring)

        self.h.create_walls()
        self.v.create_walls()
        self.wall_polygons = self.h.wall_polygons + self.v.wall_polygons

    def create_v(self, p1, p2):
        center1 = p1[0], p1[1]
        center2 = p1[0], p2[1]

        if p1[1] > p2[1]:
            temp = center1
            center1 = center2
            center2 = temp

        height = center2[1] - center1[1]
        shape = Rect_line(center1[0] - self.narrow, center1[1] - self.narrow, self.narrow + self.narrow,
                          height + self.narrow*2)
        return shape

    def create_h(self, p1, p2):
        center1 = p1[0], p1[1]
        center2 = p2[0], p1[1]

        if p1[0] > p2[0]:
            temp = center1
            center1 = center2
            center2 = temp

        width = center2[0] - center1[0]
        shape = Rect_line(center1[0] - self.narrow, center1[1] - self.narrow,
                              width + self.narrow * 2, self.narrow + self.narrow)
        return shape


class Dungeons():

    def __init__(self, space_size = (400, 600), n_rooms = 7):

        #list of all rooms
        self.rooms = []

        #walls
        self.walls = []

        #list of all coridors
        self.corridors = []

        #list of all shapes on the map (to create full topolical map later)
        shapes = []

        self.space_size = space_size
        grid_size = WALL_SIZE
        self.space_Rect = Rect(0, 0, space_size[0], space_size[1])

        max_room_size = MAX_ROOM_SIZE/grid_size
        min_room_size = MIN_ROOM_SIZE/grid_size

        space_grid = self.space_size[0] / grid_size, space_size[1] / grid_size
        all_rooms = geo.Polygon()

        for ii in range(n_rooms):
            room_width = grid_size * rand.randint(min_room_size,  max_room_size)
            room_height = grid_size * rand.randint(min_room_size, max_room_size)
            room_center = grid_size * (rand.randint(1, space_grid[0] - 1)), grid_size * (
                rand.randint(1, space_grid[1] - 1))
            new_room = Room(room_center, room_width, room_height)

            dist = new_room.ring.distance(all_rooms)
            out_of_bounds = not self.space_Rect.contains(new_room)
            while self.rooms and dist<WALL_SIZE or out_of_bounds:
                room_width = grid_size * rand.randint(min_room_size, max_room_size)
                room_height = grid_size * rand.randint(min_room_size, max_room_size)
                room_center = grid_size * (rand.randint(1, space_grid[0] - 1)), grid_size* (rand.randint(1, space_grid[1] - 1))
                new_room = Room(room_center, room_width, room_height)
                #keys = new_room.collidelistall(self.rooms)
                dist = new_room.ring.distance(all_rooms)
                out_of_bounds = not self.space_Rect.contains(new_room)

            #create corridors
            for room in self.rooms:
                corridor = Corridor(new_room, room)
                self.corridors.append(corridor)
                new_room.corridors.append(self.corridors[-1])
                room.corridors.append(self.corridors[-1])
                shapes.append(corridor.ring)

            self.rooms.append(new_room)
            shapes.append(new_room.ring)
            all_rooms = all_rooms.union(new_room.ring)

        # remove the corridors that intersect with rooms
        # TODO: Attention: the corridors are only being removed from the full list of corridors in self, but not from the lists that are coresponding to each of the rooms
        # TODO: Clean up, it can be easier!
        new_corridors = []
        for corridor in self.corridors:
            own_rooms = corridor.room_h.ring.union(corridor.room_v.ring)
            own_room_walls = corridor.room_h.wall_polygons + corridor.room_v.wall_polygons
            current_all_rooms = all_rooms.difference(own_rooms)
            if current_all_rooms.is_empty or (corridor.ring.distance(current_all_rooms) > 10) :
                walls_own_rooms = geo.Polygon()
                for wall in own_room_walls:
                    walls_own_rooms = walls_own_rooms.union(wall)
                if walls_own_rooms.intersection(corridor.ring).area <=2*CORRIDOR_SIZE*CORRIDOR_SIZE:
                    new_corridors.append(corridor)

        self.corridors = new_corridors
        self.corridors = [x for x in self.corridors if not x.ring.intersection(all_rooms).length > Corridor.narrow*4]

        #create topological map
        self.topology = self.rooms[0].ring
        for room in self.rooms:
            self.topology = self.topology.union(room.ring)

        for jj in range(len(self.corridors)):
            self.corridors[jj].ring= self.corridors[jj].ring.difference(self.topology)

        for jj in range(len(self.corridors)):
             self.topology = self.topology.union(self.corridors[jj].ring)

        walls = []
        #create walls
        for jj in range(len(self.rooms)):
            for ii in range(len(self.rooms[jj].wall_polygons)):
                self.rooms[jj].wall_polygons[ii] = self.rooms[jj].wall_polygons[ii].difference(self.topology)
                #wall = Wall(self.rooms[jj].wall_polygons[ii].boundary)
                walls.append((self.rooms[jj].wall_polygons[ii], self.rooms[jj].texture))

        for jj in range(len(self.corridors)):
            for ii in range(len(self.corridors[jj].wall_polygons)):

                    self.corridors[jj].wall_polygons[ii] = self.corridors[jj].wall_polygons[ii].difference(self.topology)
                    #print(self.corridors[jj].wall_polygons[ii])
                    if not self.corridors[jj].wall_polygons[ii].is_empty:
                        walls.append((self.corridors[jj].wall_polygons[ii], self.corridors[jj].texture))
                        #draw_multilinestring(self.corridors[jj].wall_polygons[ii].boundary, screen)

        print(len(walls))
        walls__ = convert_to_list_of_poly(walls)
        print(len(walls__))

        for wall in walls__:
            print (wall)
            # print (wall[1])
            # wall_ = Wall(wall[0].boundary, texture=wall[1])
            self.walls.append(wall)
        #TODO: check connectivity: this is implemented on the level of the environment, but should be moved here

    def generate_random_point(self, n_points=1, pymunk_coordinates = True):
        #fundtion to generate random point inside of the map (eg for the fruits)
        ll = []
        bb = self.topology.bounds
        for ii in range(n_points):
            p = (rand.randint(bb[0], bb[2]), rand.randint(bb[1], bb[3]))
            while not self.topology.contains(geo.Point(p)):
                p = (rand.randint(bb[0], bb[2]), rand.randint(bb[1], bb[3]))

            if pymunk_coordinates:
                ll.append((p[0], self.space_size[1] - p[1]))
            else:
                ll.append((p[0], p[1]))
        return ll

    def draw(self):
        #standalone display of the topological elements of the map using pygame engine
        #this is mainly needed for debugging, hence not removing old code
        pygame.init()
        screen = pygame.display.set_mode(self.space_size)

        #for room in self.rooms:
            #pygame.draw.rect(screen, (0, 255, 255), room, 2)

        # for corridor in self.corridors:
        #     pygame.draw.rect(screen, (0, 255, 255), corridor.h, 2)
        #     pygame.draw.rect(screen, (0, 255, 255), corridor.v, 2)

        # for corridor in self.corridors:
        #     print(corridor.ring.boundary)
        #     array_of_points = np.array(corridor.ring.boundary)
        #     for ii in range(1, array_of_points.shape[0]):
        #        pygame.draw.line(screen, (255, 255, 0), array_of_points[ii-1, :], array_of_points[ii, :], 2)
        #        print(array_of_points[ii-1, :], array_of_points[ii, :])
        #     pygame.draw.line(screen, (255, 255, 0), array_of_points[-1, :], array_of_points[0, :], 2)
        #     # # for ii in range(1, array_of_points.shape[0]):
        #     #     pygame.draw.line(screen, (255, 255, 0), array_of_points[ii-1, :], array_of_points[ii, :], 2)
        #     #     print(array_of_points[ii-1, :], array_of_points[ii, :])
        #     # pygame.draw.line(screen, (255, 255, 0), array_of_points[-1, :], array_of_points[0, :], 2)
        #
        #     # for x in np.nditer(array_of_points):
        #     #     pygame.draw.line(screen, (0, 255, 0), line_[x, :], line_[1, :], 2)
        #     # #pygame.draw.rect(screen, (255, 0, 0), corridor.v, 2)
        #     #pygame.draw.rect(screen, (255, 0, 0), corridor.h, 2)

        draw_multilinestring(self.topology.boundary, screen)

        #for room in self.rooms:
            #draw_linearring(room.outer_line.boundary, screen)

        # for jj in range(len(self.rooms)):
        #     for ii in range(len(self.rooms[jj].wall_polygons)):
        #         self.rooms[jj].wall_polygons[ii] = self.rooms[jj].wall_polygons[ii].difference(self.topology)
        #         draw_multilinestring(self.rooms[jj].wall_polygons[ii].boundary, screen)
        #
        #
        # for jj in range(len(self.corridors)):
        #     for ii in range(len(self.corridors[jj].wall_polygons)):
        #
        #             self.corridors[jj].wall_polygons[ii] = self.corridors[jj].wall_polygons[ii].difference(self.topology)
        #             print(self.corridors[jj].wall_polygons[ii])
        #             if not self.corridors[jj].wall_polygons[ii].is_empty:
        #                 draw_multilinestring(self.corridors[jj].wall_polygons[ii].boundary, screen)
        #

        for wall in self.walls:
            pygame.draw.rect(screen, (0, 255, 255), Rect(wall.x - 0.5 * wall.width, wall.y - 0.5 * wall.height, wall.width, wall.height ), 2)
            pygame.draw.circle(screen, (255, 0, 0), self.generate_random_point(pymunk_coordinates=False)[-1], 5, 1 )
        # print(self.topology.boundary)
        # sh = geo.shape(self.topology.boundary)
        # for line in sh:
        #     array_of_points = np.array(line)
        #     for ii in range(1, array_of_points.shape[0]):
        #         pygame.draw.line(screen, (255, 255, 0), array_of_points[ii - 1, :], array_of_points[ii, :], 2)
        #         print(array_of_points[ii - 1, :], array_of_points[ii, :])
        #     pygame.draw.line(screen, (255, 255, 0), array_of_points[-1, :], array_of_points[0, :], 2)


        # for line in sh:
        #     print (np.array(line))
        #     line_ = np.array(line)
        #     pygame.draw.line(screen, (0, 255, 0), line_[0, :], line_[1, :], 2)

        pygame.display.update()

        clock = pygame.time.Clock()

        # pygame.display.update()
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            screen.fill((128, 0, 128))

            clock.tick(20)


if __name__ == "__main__":
    mapp = Dungeons()
    mapp.draw()