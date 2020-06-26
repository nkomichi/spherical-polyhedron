import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import SphericalVoronoi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def distance(point_from, point_to):
    return math.sqrt((point_from[0] - point_to[0]) ** 2
                     + (point_from[1] - point_to[1]) ** 2
                     + (point_from[2] - point_to[2]) ** 2)


class SphericalPolyhedron():
    """球のクラス"""
    def __init__(self, center=np.array([0.0, 0.0, 0.0]), radius=1.0, points=0):
        self.center, self.radius = center, float(radius)
        self.points = np.empty((0, 3))
        self.add_random_point(0)
        self.step = 0

    def plot(self, elev=30, azim=30):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color='lightskyblue', linewidth=0.5)
        crds = self.points.T
        ax.plot(crds[0], crds[1], crds[2], marker='o', linestyle='None',)
        ax.view_init(elev=elev, azim=azim)
        plt.show()

    def add_random_point(self, number=1):
        """numberの数だけ球面上にランダムな点を加える"""
        for _ in range(number):
            # theta = random.random() * math.pi
            phi = random.random() * math.pi * 2
            z = random.random() * 2 - 1
            self.points = np.append(
                self.points,
                [[self.radius * math.sqrt(1 - (z ** 2)) * math.cos(phi),
                  self.radius * math.sqrt(1 - (z ** 2)) * math.sin(phi),
                  self.radius * z]],
                axis=0)

    def bind_point_on_surface(self, point: np.ndarray) -> np.ndarray:
        """点を受け取りその点を球面上に落とした点を返す"""
        factor = self.radius / distance(point, self.center)
        return point * factor

    def bind_all_points(self):
        """全ての点を球面上に落とす"""
        for i, point in enumerate(self.points):
            self.points[i] = self.bind_point_on_surface(point)

    def bind_on_surface(self):
        distances = np.linalg.norm(self.points - self.center, axis=1, keepdims=True)
        self.points = self.points / distances * self.radius

    def snap_to_center(self):
        """全点の重心を球の中心と一致させる"""
        self.points += self.center - self.points.mean(axis=0)

    def proceed(self, constant=-1, power=2, cutoff=2.0, snap=True, bind=True):
        """全ての点に対し距離のpower乗に反比例し、constantが正なら引力、負なら斥力"""
        new_points = np.empty([len(self.points), 3])
        for i, point in enumerate(self.points):
            vector = np.array([0.0, 0.0, 0.0])
            for target in self.points:
                if (point == target).all():
                    continue
                v = target - point
                d = np.linalg.norm(v)
                # nv = v / d
                if d < cutoff:
                    vector += v * constant / (d ** power)
            new_points[i] = point + vector
        logger.debug("new_points=%s", new_points)
        self.points = new_points
        if snap:
            self.snap_to_center()
        if bind:
            self.bind_on_surface()
        self.step += 1

    def voronoi(self, plot=False):
        """球面ボロノイによる評価"""
        areas = []
        points = self.points
        center = self.center
        sv = SphericalVoronoi(points, self.radius, center)
        # 面積を計算
        for region in sv.regions:
            triangles = []
            number_of_triangles = len(region) - 2
            for n in range(number_of_triangles):
                triangles.append([region[0], region[n+1], region[n+2]])
            angle = 0
            for triangle in triangles:
                OO = self.center
                A = sv.vertices[triangle[0]] - OO
                B = sv.vertices[triangle[1]] - OO
                C = sv.vertices[triangle[2]] - OO
                cAB = np.cross(A, B)
                cAB = cAB / np.linalg.norm(cAB)
                cBC = np.cross(B, C)
                cBC = cBC / np.linalg.norm(cBC)
                cCA = np.cross(C, A)
                cCA = cCA / np.linalg.norm(cCA)
                ABBC = max(-1.0, min(1.0, np.dot(cAB, -cBC)))
                BCCA = max(-1.0, min(1.0, np.dot(cBC, -cCA)))
                CAAB = max(-1.0, min(1.0, np.dot(cCA, -cAB)))
                a = 0
                a += math.acos(ABBC)
                a += math.acos(BCCA)
                a += math.acos(CAAB)
                a -= math.pi
                angle += a
            areas.append(angle)

        if plot:
            # sort vertices (optional, helpful for plotting)
            sv.sort_vertices_of_regions()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # plot the unit sphere for reference (optional)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='y', alpha=0.1)
            # plot generator points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
            # plot Voronoi vertices
            ax.scatter(
                sv.vertices[:, 0],
                sv.vertices[:, 1],
                sv.vertices[:, 2],
                c='g')
            # indicate Voronoi regions (as Euclidean polygons)
            for region in sv.regions:
                random_color = colors.rgb2hex(np.random.rand(3))
                polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
                polygon.set_color(random_color)
                ax.add_collection3d(polygon)
            plt.show()
        return areas

    def edges(self):
        edges = []
        for i, point in enumerate(self.points):
            for target in self.points[i+1:]:
                edges.append(np.linalg.norm(target - point))
        return edges

    def save(self, filename='', directory=''):
        if filename == '':
            filename = f'{len(self.points)}_{self.step}.points'
        if directory:
            filename = directory + filename
        np.savetxt(filename, self.points)
