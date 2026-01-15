from __future__ import annotations
from .geom import *
import deepsvg.svglib.geom as geom
import re
import torch
from typing import List, Union
from xml.dom import minidom
import math
import shapely.geometry
import numpy as np
import logging
from .geom import union_bbox
#from .svg_command import SVGCommand, SVGCommandMove, SVGCommandClose, SVGCommandBezier, SVGCommandLine, SVGCommandArc, SVGCommandColor
from .svg_command import SVGCommand, SVGCommandMove, SVGCommandClose, SVGCommandBezier, SVGCommandLine, SVGCommandArc

COMMANDS = "MmZzLlHhVvCcSsQqTtAa"
COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


empty_command = SVGCommandMove(Point(0.))

class Orientation:
    COUNTER_CLOCKWISE = 0
    CLOCKWISE = 1


class Filling:
    OUTLINE = 0
    FILL = 1
    ERASE = 2


class SVGPath:
    def __init__(self, path_commands: List[SVGCommand] = None, origin: Point = None, closed=False,
                 filling=Filling.OUTLINE, fill="none", stroke="white", stroke_width=1.0, dasharray=None,
                 fill_opacity=0.0, stroke_opacity=1.0, fill_rule = "evenodd"):
        self.origin = origin or Point(0.)
        self.path_commands = path_commands
        self.closed = closed

        self.filling = filling
        self.fill = fill  # Fill color 
        self.stroke = stroke  # Stroke color
        self.stroke_width = stroke_width  # Stroke width
        self.stroke_opacity = stroke_opacity
        self.dasharray = dasharray  # Dash array for stroke patterns
        self.fill_opacity = fill_opacity  # Overall opacity
        self.fill_rule=fill_rule

    @property
    def start_command(self):
        return SVGCommandMove(self.origin, self.start_pos)

    @property
    def start_pos(self):
        return self.path_commands[0].start_pos

    @property
    def end_pos(self):
        return self.path_commands[-1].end_pos

    def to_group(self, *args, **kwargs):
        from .svg_primitive import SVGPathGroup
        return SVGPathGroup(
            [self],
            fill=self.fill,
            fill_opacity=self.fill_opacity,
            fill_rule=self.fill_rule,
            stroke=self.stroke,
            stroke_width=self.stroke_width,
            stroke_opacity=self.stroke_opacity,
            dasharray=self.dasharray,
            filling=self.filling
        )
        #return SVGPathGroup([self], *args, **kwargs)

    def set_filling(self, filling=True):
        self.filling = Filling.FILL if filling else Filling.ERASE
        return self

    def __len__(self):
        return 1 + len(self.path_commands) + int(self.closed)

    def __getitem__(self, idx):
        if idx == 0:
            return self.start_command
        return self.path_commands[idx-1]

    def all_commands(self, with_close=True):
        close_cmd = [SVGCommandClose(self.path_commands[-1].end_pos.copy(), self.start_pos.copy())] if self.closed and self.path_commands and with_close \
                    else ()
        return [self.start_command, *self.path_commands, *close_cmd]

    def copy(self):
        return SVGPath(
            path_commands=[cmd.copy() for cmd in self.path_commands],
            origin=self.origin.copy(),
            closed=self.closed,
            filling=self.filling,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
            fill_rule=self.fill_rule,
            stroke=self.stroke,
            stroke_width=self.stroke_width,
            stroke_opacity=self.stroke_opacity,
            dasharray=self.dasharray
        )
        #return SVGPath([path_command.copy() for path_command in self.path_commands], self.origin.copy(), self.closed, filling=self.filling)

    @staticmethod
    def _tokenize_path(path_str):
        cmd = None
        for x in COMMAND_RE.split(path_str):
            if x and x in COMMANDS:
                cmd = x
            elif cmd is not None:
                yield cmd, list(map(float, FLOAT_RE.findall(x)))

    @staticmethod
    def from_xml(x: minidom.Element):
        fill = x.getAttribute("fill") if x.hasAttribute("fill") else "none"
        fill_opacity = float(x.getAttribute("fill-opacity")) if x.hasAttribute("fill-opacity") else 1.0
        if fill_opacity == 0.0:
            fill_opacity = 1.0

        fill_rule = x.getAttribute("fill-rule") or 'nonzero'
        path_fill_rule = fill_rule


        stroke = x.getAttribute("stroke") or None
        stroke_opacity = float(x.getAttribute("stroke-opacity")) if x.hasAttribute("stroke-opacity") else 1.0
        stroke_width = float(x.getAttribute("stroke-width")) if x.hasAttribute("stroke-width") else 3.0


        s = x.getAttribute('d')
        svg_path = SVGPath.from_str(s, fill=fill, stroke=stroke, 
                                fill_opacity=fill_opacity, stroke_opacity=stroke_opacity, stroke_width=stroke_width)
        svg_path.fill_rule = path_fill_rule
        return svg_path


    
    @staticmethod
    def from_str(s: str, fill="none", fill_opacity=0.0, fill_rule = "evenodd", stroke="black", stroke_width="1.0", stroke_opacity=1.0, dasharray=None,  filling=Filling.OUTLINE, add_closing=False):
        path_commands = []
        pos = initial_pos = Point(0.)
        prev_command = None

        # Tokenize and parse the SVG path commands
        for cmd, args in SVGPath._tokenize_path(s):
            cmd_parsed, pos, initial_pos = SVGCommand.from_str(cmd, args, pos, initial_pos, prev_command)
            prev_command = cmd_parsed[-1]
            path_commands.extend(cmd_parsed)
        
        #color_token = color_to_token(fill)
        #color_command = SVGCommandColor(color_token)
        #path_commands.append(color_command)
        # Pass the additional attributes such as fill, stroke, and opacity to the SVGPath.from_commands method
        return SVGPath.from_commands(
            path_commands, 
            fill=fill, 
            fill_opacity=fill_opacity,
            fill_rule = fill_rule,
            stroke=stroke, 
            stroke_width=stroke_width, 
            stroke_opacity=stroke_opacity,
            dasharray=dasharray, 
            filling=filling, 
            add_closing=add_closing,
        )


    @staticmethod
    def from_tensor(tensor: torch.Tensor, allow_empty=False):
        return SVGPath.from_commands([SVGCommand.from_tensor(row) for row in tensor], allow_empty=allow_empty)

    @staticmethod
    def from_commands(path_commands: List[SVGCommand], fill=False, stroke="black", stroke_width="1.0", dasharray=None, fill_opacity=0.0, fill_rule="evenodd", stroke_opacity=1.0, filling=Filling.OUTLINE, add_closing=False, allow_empty=False ):
        from .svg_primitive import SVGPathGroup

        if not path_commands:
            return SVGPathGroup([])

        svg_paths = []
        svg_path = None

        for command in path_commands:
            if isinstance(command, SVGCommandMove):
                if svg_path is not None and (allow_empty or svg_path.path_commands):  # SVGPath contains at least one command
                    if add_closing:
                        svg_path.closed = True
                    if not svg_path.path_commands:
                        svg_path.path_commands.append(empty_command)
                    svg_paths.append(svg_path)

                # 创建新的 SVGPath，并传递颜色相关属性
                svg_path = SVGPath(
                    [], 
                    command.start_pos.copy(), 
                    filling=filling,
                    fill=fill,
                    fill_opacity=fill_opacity,
                    fill_rule=fill_rule,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    stroke_opacity=stroke_opacity,
                    dasharray=dasharray,
                )
                svg_path.path_commands.append(command)
                #print("SVGCommandMove detected and added to path_commands.")

            else:
                if svg_path is None:
                    # Ignore commands until the first moveTo commands
                    continue

                if isinstance(command, SVGCommandClose):
                    #if allow_empty or svg_path.path_commands:  # SVGPath contains at least one command
                    if svg_path is not None and (allow_empty or svg_path.path_commands):
                        svg_path.closed = True
                        svg_path.path_commands.append(command)
                        #print("SVGCommandClose detected and added to path_commands.")

                        if command.start_pos != command.end_pos:
                            svg_path.path_commands.append(SVGCommandLine(command.start_pos.copy(), command.end_pos.copy()))
                        if not svg_path.path_commands:
                            svg_path.path_commands.append(empty_command)
                        svg_paths.append(svg_path)
                    svg_path = None
                else:
                    if svg_path is None:
                        # Ignore commands until the first Move command
                        continue
                    svg_path.path_commands.append(command)
                    #print(f"Command {type(command)} added to path_commands.")

        if svg_path is not None and (allow_empty or svg_path.path_commands):  # SVGPath contains at least one command
            if add_closing:
                svg_path.closed = True
            if not svg_path.path_commands:
                svg_path.path_commands.append(empty_command)
            svg_paths.append(svg_path)
            #print("Final SVGPath added to SVGPathGroup.")

        return SVGPathGroup(svg_paths, color=fill, fill=fill, fill_opacity=fill_opacity, fill_rule=fill_rule,stroke_opacity=stroke_opacity, stroke=stroke, stroke_width=stroke_width, dasharray=dasharray)


    def __repr__(self):
        return "SVGPath({})".format(" ".join(command.__repr__() for command in self.all_commands()))

    def to_str(self, fill=False):
        return " ".join(command.to_str() for command in self.all_commands())
    
    def to_tensor(self, PAD_VAL=0):
        return torch.stack([command.to_tensor(PAD_VAL=PAD_VAL) for command in self.all_commands()])


    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=False, with_moves=True):
        points = self._get_points_viz(color_firstlast, with_moves) if with_points else ()
        handles = self._get_handles_viz() if with_handles else ()
        return [*points, *handles]

    '''
    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG
        return SVG([self.to_group()], viewbox=viewbox).draw(*args, **kwargs)
    '''
    def draw(self, ax, fill=False, stroke_color='black', stroke_width=1.0, fill_color='black', fill_opacity=1.0, rotation=0):
        import matplotlib.patches as patches
        import matplotlib.path as mpath
        import numpy as np
        
        path_data = self.get_matplotlib_path_data()
        path = mpath.Path(path_data['vertices'], path_data['codes'])
        transform = patches.Affine2D().rotate_deg(rotation) + ax.transData

        # 设置填充颜色和透明度
        patch = patches.PathPatch(
            path,
            facecolor=fill_color if fill else 'none',
            edgecolor=stroke_color,
            lw=stroke_width,
            alpha=fill_opacity  # 设置填充透明度
        )
        patch.set_transform(transform)
        ax.add_patch(patch)

    def _get_points_viz(self, color_firstlast=True, with_moves=True):
        points = []
        commands = self.all_commands(with_close=False)
        n = len(commands)
        for i, command in enumerate(commands):
            if not isinstance(command, SVGCommandMove) or with_moves:
                points_viz = command.get_points_viz(first=(color_firstlast and i <= 1), last=(color_firstlast and i >= n-2))
                points.extend(points_viz)
        return points

    def _get_handles_viz(self):
        handles = []
        for command in self.path_commands:
            handles.extend(command.get_handles_viz())
        return handles

    def _get_unique_geoms(self):
        geoms = []
        for command in self.all_commands():
            geoms.extend(command.get_geoms())
        return list(set(geoms))

    def translate(self, vec):
        for geom in self._get_unique_geoms():
            geom.translate(vec)
        return self

    def rotate(self, angle):
        for geom in self._get_unique_geoms():
            geom.rotate_(angle)
        return self

    def scale(self, factor):
        for geom in self._get_unique_geoms():
            geom.scale(factor)
        return self

    def filter_consecutives(self):
        path_commands = []
        for command in self.path_commands:
            if not command.start_pos.isclose(command.end_pos):
                path_commands.append(command)
        self.path_commands = path_commands
        return self
    '''
    def filter_duplicates(self, min_dist=0.2):
        path_commands = []
        current_command = None
        for command in self.path_commands:
            if current_command is None:
                path_commands.append(command)
                current_command = command

            if command.end_pos.dist(current_command.end_pos) >= min_dist:
                command.start_pos = current_command.end_pos
                path_commands.append(command)
                current_command = command

        self.path_commands = path_commands
        return self
    '''

    def filter_duplicates(self, min_dist=0.2, close_path=False):
        path_commands = []
        current_command = None

        for command in self.path_commands:
            if current_command is None:
                # Start the path with the first command
                path_commands.append(command)
                current_command = command
                continue

            # Calculate the distance between the current command's end position and the last saved command's end position
            if command.end_pos.dist(current_command.end_pos) >= min_dist:
                command.start_pos = current_command.end_pos
                path_commands.append(command)
                current_command = command
            else:
                # If the distance is less than min_dist, skip the command (likely a duplicate or unnecessary short segment)
                continue

        # Check if path needs to be closed
        if close_path:
            if path_commands:
                first_command = path_commands[0]
                last_command = path_commands[-1]
            
                # If the last command's end position is not the same as the first command's start position, close the path
                if last_command.end_pos.dist(first_command.start_pos) >= min_dist:
                    close_command = SVGCommandLine(start_pos=last_command.end_pos, end_pos=first_command.start_pos)
                    path_commands.append(close_command)

        self.path_commands = path_commands
        return self


    def duplicate_extremities(self):
        self.path_commands = [SVGCommandLine(self.start_pos, self.start_pos),
                              *self.path_commands,
                              SVGCommandLine(self.end_pos, self.end_pos)]
        return self

    def is_clockwise(self):
        if len(self.path_commands) == 1:
            cmd = self.path_commands[0]
            return cmd.start_pos.tolist() <= cmd.end_pos.tolist()

        det_total = 0.
        for cmd in self.path_commands:
            det_total += geom.det(cmd.start_pos, cmd.end_pos)
        return det_total >= 0.

    def set_orientation(self, orientation):
        """
        orientation: 1 (clockwise), 0 (counter-clockwise)
        """
        if orientation == self.is_clockwise():
            return self
        return self.reverse()

    def set_closed(self, closed=True):
        self.closed = closed
        return self

    def reverse(self):
        path_commands = []

        for command in reversed(self.path_commands):
            path_commands.append(command.reverse())

        self.path_commands = path_commands
        return self

    def reverse_non_closed(self):
        if not self.start_pos.isclose(self.end_pos):
            return self.reverse()
        return self

    def simplify_arcs(self):
        path_commands = []
        for command in self.path_commands:
            if isinstance(command, SVGCommandArc):
                if command.radius.iszero():
                    continue
                if command.start_pos.isclose(command.end_pos):
                    continue
                path_commands.extend(command.to_beziers())
            else:
                path_commands.append(command)

        self.path_commands = path_commands
        return self

    def _get_topleftmost_command(self):
        topleftmost_cmd = None
        topleftmost_idx = 0

        for i, cmd in enumerate(self.path_commands):
            if topleftmost_cmd is None or cmd.is_left_to(topleftmost_cmd):
                topleftmost_cmd = cmd
                topleftmost_idx = i

        return topleftmost_cmd, topleftmost_idx

    def reorder(self):
        # if self.closed:
        topleftmost_cmd, topleftmost_idx = self._get_topleftmost_command()

        self.path_commands = [
            *self.path_commands[topleftmost_idx:],
            *self.path_commands[:topleftmost_idx]
        ]

        return self

    def to_video(self, wrapper, clips=None, svg_commands=None, color="grey"):
        from .svg import SVG
        from .svg_primitive import SVGLine, SVGCircle

        if clips is None:
            clips = []
        if svg_commands is None:
            svg_commands = []
        svg_dots, svg_moves = [], []

        for command in self.all_commands():
            start_pos, end_pos = command.start_pos, command.end_pos

            if isinstance(command, SVGCommandMove):
                move = SVGLine(start_pos, end_pos, color="teal", dasharray=0.5)
                svg_moves.append(move)

            dot = SVGCircle(end_pos, radius=Radius(0.1), color="red")
            svg_dots.append(dot)

            svg_path = SVGPath(svg_commands).to_group(color=color)
            svg_new_path = SVGPath([SVGCommandMove(start_pos), command]).to_group(color="red")

            svg_paths = [svg_path, svg_new_path]  if svg_commands else [svg_new_path]
            im = SVG([*svg_paths, *svg_moves, *svg_dots]).draw(do_display=False, return_png=True, with_points=False)
            clips.append(wrapper(np.array(im)))

            svg_dots[-1].color = "grey"
            svg_commands.append(command)
            svg_moves = []

        return clips, svg_commands

    def numericalize(self, n=256):
        for command in self.all_commands():
            command.numericalize(n)

    def smooth(self):
        # https://github.com/paperjs/paper.js/blob/c7d85b663edb728ec78fffa9f828435eaf78d9c9/src/path/Path.js#L1288
        n = len(self.path_commands)
        knots = [self.start_pos, *(path_commmand.end_pos for path_commmand in self.path_commands)]
        r = [knots[0] + 2 * knots[1]]
        f = [2]
        p = [Point(0.)] * (n + 1)

        # Solve with the Thomas algorithm
        for i in range(1, n):
            internal = i < n - 1
            a = 1
            b = 4 if internal else 2
            u = 4 if internal else 3
            v = 2 if internal else 0
            m = a / f[i-1]

            f.append(b-m)
            r.append(u * knots[i] + v * knots[i + 1] - m * r[i-1])

        p[n-1] = r[n-1] / f[n-1]
        for i in range(n-2, -1, -1):
            p[i] = (r[i] - p[i+1]) / f[i]
        p[n] = (3 * knots[n] - p[n-1]) / 2

        for i in range(n):
            p1, p2 = knots[i], knots[i+1]
            c1, c2 = p[i], 2 * p2 - p[i+1]
            self.path_commands[i] = SVGCommandBezier(p1, c1, c2, p2)

        return self

    def simplify_heuristic(self):
        return self.copy().split(max_dist=2, include_lines=False) \
            .simplify(tolerance=0.1, epsilon=0.2, angle_threshold=150) \
            .split(max_dist=7.5)

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        # https://github.com/paperjs/paper.js/blob/c044b698c6b224c10a7747664b2a4cd00a416a25/src/path/PathFitter.js#L44
        points = [self.start_pos, *(path_command.end_pos for path_command in self.path_commands)]

        def subdivide_indices():
            segments_list = []
            current_segment = []
            prev_command = None

            for i, command in enumerate(self.path_commands):
                if isinstance(command, SVGCommandLine):
                    if current_segment:
                        segments_list.append(current_segment)
                        current_segment = []
                    prev_command = None

                    continue

                if prev_command is not None and prev_command.angle(command) < angle_threshold:
                    if current_segment:
                        segments_list.append(current_segment)
                        current_segment = []

                current_segment.append(i)
                prev_command = command

            if current_segment:
                segments_list.append(current_segment)

            return segments_list

        path_commands = []

        def computeMaxError(first, last, curve: SVGCommandBezier, u):
            maxDist = 0.
            index = (last - first + 1) // 2
            for i in range(1, last - first):
                dist = curve.eval(u[i]).dist(points[first + i]) ** 2
                if dist >= maxDist:
                    maxDist = dist
                    index = first + i
            return maxDist, index

        def chordLengthParametrize(first, last):
            u = [0.]
            for i in range(1, last - first + 1):
                u.append(u[i-1] + points[first + i].dist(points[first + i-1]))

            for i, _ in enumerate(u[1:], 1):
                u[i] /= u[-1]

            return u

        def isMachineZero(val):
            MACHINE_EPSILON = 1.12e-16
            return val >= -MACHINE_EPSILON and val <= MACHINE_EPSILON

        def findRoot(curve: SVGCommandBezier, point, u):
            """
               Newton's root finding algorithm calculates f(x)=0 by reiterating
               x_n+1 = x_n - f(x_n)/f'(x_n)
               We are trying to find curve parameter u for some point p that minimizes
               the distance from that point to the curve. Distance point to curve is d=q(u)-p.
               At minimum distance the point is perpendicular to the curve.
               We are solving
               f = q(u)-p * q'(u) = 0
               with
               f' = q'(u) * q'(u) + q(u)-p * q''(u)
               gives
               u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
            """
            diff = curve.eval(u) - point
            d1, d2 = curve.derivative(u, n=1), curve.derivative(u, n=2)
            numerator = diff.dot(d1)
            denominator = d1.dot(d1) + diff.dot(d2)

            return u if isMachineZero(denominator) else u - numerator / denominator

        def reparametrize(first, last, u, curve: SVGCommandBezier):
            for i in range(0, last - first + 1):
                u[i] = findRoot(curve, points[first + i], u[i])

            for i in range(1, len(u)):
                if u[i] <= u[i-1]:
                    return False

            return True

        def generateBezier(first, last, uPrime, tan1, tan2):
            epsilon = 1e-12
            p1, p2 = points[first], points[last]
            C = np.zeros((2, 2))
            X = np.zeros(2)

            for i in range(last - first + 1):
                u = uPrime[i]
                t = 1 - u
                b = 3 * u * t
                b0 = t**3
                b1 = b * t
                b2 = b * u
                b3 = u**3
                a1 = tan1 * b1
                a2 = tan2 * b2
                tmp = points[first + i] - p1 * (b0 + b1) - p2 * (b2 + b3)

                C[0, 0] += a1.dot(a1)
                C[0, 1] += a1.dot(a2)
                C[1, 0] = C[0, 1]
                C[1, 1] += a2.dot(a2)
                X[0] += a1.dot(tmp)
                X[1] += a2.dot(tmp)

            detC0C1 = C[0, 0] * C[1, 1] - C[1, 0] * C[0, 1]
            if abs(detC0C1) > epsilon:
                detC0X = C[0, 0] * X[1] - C[1, 0] * X[0]
                detXC1 = X[0] * C[1, 1] - X[1] * C[0, 1]
                alpha1 = detXC1 / detC0C1
                alpha2 = detC0X / detC0C1
            else:
                c0 = C[0, 0] + C[0, 1]
                c1 = C[1, 0] + C[1, 1]
                alpha1 = alpha2 = X[0] / c0 if abs(c0) > epsilon else (X[1] / c1 if abs(c1) > epsilon else 0)

            segLength = p2.dist(p1)
            eps = epsilon * segLength
            handle1 = handle2 = None

            if alpha1 < eps or alpha2 < eps:
                alpha1 = alpha2 = segLength / 3
            else:
                line = p2 - p1
                handle1 = tan1 * alpha1
                handle2 = tan2 * alpha2

                if handle1.dot(line) - handle2.dot(line) > segLength**2:
                    alpha1 = alpha2 = segLength / 3
                    handle1 = handle2 = None

            if handle1 is None or handle2 is None:
                handle1 = tan1 * alpha1
                handle2 = tan2 * alpha2

            return SVGCommandBezier(p1, p1 + handle1, p2 + handle2, p2)

        def computeLinearMaxError(first, last):
            maxDist = 0.
            index = (last - first + 1) // 2

            p1, p2 = points[first], points[last]
            for i in range(first + 1, last):
                dist = points[i].distToLine(p1, p2)
                if dist >= maxDist:
                    maxDist = dist
                    index = i
            return maxDist, index

        def ramerDouglasPeucker(first, last, epsilon):
            max_error, split_index = computeLinearMaxError(first, last)

            if max_error > epsilon:
                ramerDouglasPeucker(first, split_index, epsilon)
                ramerDouglasPeucker(split_index, last, epsilon)
            else:
                p1, p2 = points[first], points[last]
                path_commands.append(SVGCommandLine(p1, p2))

        def fitCubic(error, first, last, tan1=None, tan2=None):
            # For convenience, compute extremity tangents if not provided
            if tan1 is None and tan2 is None:
                tan1 = (points[first + 1] - points[first]).normalize()
                tan2 = (points[last - 1] - points[last]).normalize()

            if last - first == 1:
                p1, p2 = points[first], points[last]
                dist = p1.dist(p2) / 3
                path_commands.append(SVGCommandBezier(p1, p1 + dist * tan1, p2 + dist * tan2, p2))
                return

            uPrime = chordLengthParametrize(first, last)
            maxError = max(error, error**2)
            parametersInOrder = True

            for i in range(5):
                curve = generateBezier(first, last, uPrime, tan1, tan2)

                max_error, split_index = computeMaxError(first, last, curve, uPrime)

                if max_error < error and parametersInOrder:
                    path_commands.append(curve)
                    return

                if max_error >= maxError:
                    break

                parametersInOrder = reparametrize(first, last, uPrime, curve)
                maxError = max_error

            tanCenter = (points[split_index-1] - points[split_index+1]).normalize()
            fitCubic(error, first, split_index, tan1, tanCenter)
            fitCubic(error, split_index, last, -tanCenter, tan2)

        segments_list = subdivide_indices()
        if force_smooth:
            fitCubic(tolerance, 0, len(points) - 1)
        else:
            if segments_list:
                seg = segments_list[0]
                ramerDouglasPeucker(0, seg[0], epsilon)

                for seg, seg_next in zip(segments_list[:-1], segments_list[1:]):
                    fitCubic(tolerance, seg[0], seg[-1] + 1)
                    ramerDouglasPeucker(seg[-1] + 1, seg_next[0], epsilon)

                seg = segments_list[-1]
                fitCubic(tolerance, seg[0], seg[-1] + 1)
                ramerDouglasPeucker(seg[-1] + 1, len(points) - 1, epsilon)
            else:
                ramerDouglasPeucker(0, len(points) - 1, epsilon)

        self.path_commands = path_commands

        return self

    def split(self, n=None, max_dist=None, include_lines=True):
        path_commands = []

        for command in self.path_commands:
            if isinstance(command, SVGCommandLine) and not include_lines:
                path_commands.append(command)
            else:
                l = command.length()
                if max_dist is not None:
                    n = max(math.ceil(l / max_dist), 1)

                path_commands.extend(command.split(n=n))

        self.path_commands = path_commands

        return self

    def bbox(self):
        return union_bbox([cmd.bbox() for cmd in self.path_commands])

    def sample_points(self, max_dist=0.4):
        points = []

        for command in self.path_commands:
            l = command.length()
            n = max(math.ceil(l / max_dist), 1)
            sample_points = command.sample_points(n=n, return_array=True)
            if sample_points is None:
                print(f"Warning: sample_points is None for command: {command}, type: {type(command)}")
                continue  # 或者根据需要处理
            try:
                points.extend(sample_points[None])
            except Exception as e:
                print(f"Error while extending points: {e}")
                print(f"Command: {command}, sample_points: {sample_points}")
                raise
        points = np.concatenate(points, axis=0)
        return points


    def to_shapely(self):
        polygon = shapely.geometry.Polygon(self.sample_points())

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon

    def to_points(self):
        return np.array([self.start_pos.pos, *(cmd.end_pos.pos for cmd in self.path_commands)])
    
    def get_length(self):
        length = 0
        for command in self.path_commands:
            length += command.length()
        return length


    def get_matplotlib_path_data(self):
        import matplotlib.path as mpath
        import numpy as np
        from matplotlib.path import Path
        

        vertices = []
        codes = []

        current_pos = None

        for command in self.all_commands():
            if isinstance(command, SVGCommandMove):
                current_pos = command.end_pos
                vertices.append((current_pos.x, current_pos.y))
                codes.append(Path.MOVETO)
            elif isinstance(command, SVGCommandLine):
                current_pos = command.end_pos
                vertices.append((current_pos.x, current_pos.y))
                codes.append(Path.LINETO)
            elif isinstance(command, SVGCommandBezier):
                # Cubic Bezier curve
                vertices.extend([
                    (command.control1.x, command.control1.y),
                    (command.control2.x, command.control2.y),
                    (command.end_pos.x, command.end_pos.y)
                ])
                codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                current_pos = command.end_pos
            elif isinstance(command, SVGCommandArc):
                #print(f"SVGCommandArc properties: {dir(command)}")
                '''
                print(f"Types - x1: {type(x1 := current_pos.x)}, y1: {type(y1 := current_pos.y)}, "
                  f"rx: {type(rx := command.rX)}, ry: {type(ry := command.rY)}, "
                  f"phi: {type(phi := command.x_axis_rotation.degrees)}, "
                  f"large_arc_flag: {type(large_arc_flag := command.large_arc_flag.flag)}, "
                  f"sweep_flag: {type(sweep_flag := command.sweep_flag.flag)}, "
                  f"x2: {type(x2 := command.end_pos.x)}, y2: {type(y2 := command.end_pos.y)}")
                '''
                # 确认参数类型
                #print(f"Types - x1: {type(current_pos.x)}, y1: {type(current_pos.y)}, rx: {type(command.radius.rX)}, ry: {type(command.radius.rY)}, phi: {type(command.x_axis_rotation.degrees)}, large_arc_flag: {type(command.large_arc_flag.flag)}, sweep_flag: {type(command.sweep_flag.flag)}, x2: {type(command.end_pos.x)}, y2: {type(command.end_pos.y)}")
    
                beziers = arc_to_bezier(
                    x1=current_pos.x, y1=current_pos.y,
                    rx=command.rX, ry=command.rY,
                    phi=command.x_axis_rotation.degrees,
                    large_arc_flag=command.large_arc_flag.flag,
                    sweep_flag=command.sweep_flag.flag,
                    x2=command.end_pos.x, y2=command.end_pos.y
                )
                if not isinstance(beziers, list):
                    print(f"arc_to_bezier should return a list, got {type(beziers)} instead.")
                    continue

                for bezier in beziers:
                    # bezier 是包含四个点的列表 [start, control1, control2, end]
                    if not isinstance(bezier, list) or len(bezier) != 4:
                        print(f"Each bezier should be a list of 4 points, got {bezier} instead.")
                        continue
                    vertices.extend([
                        (bezier[1][0], bezier[1][1]),
                        (bezier[2][0], bezier[2][1]),
                        (bezier[3][0], bezier[3][1])
                    ])
                    codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                current_pos = command.end_pos
            elif isinstance(command, SVGCommandClose):
                codes.append(Path.CLOSEPOLY)
                vertices.append((current_pos.x, current_pos.y))
            else:
                # 对于未处理的命令，您可以选择忽略或抛出异常
                print(f"Unknown command type: {type(command)}")
                pass

        vertices = np.array(vertices, float)
        return {'vertices': vertices, 'codes': codes}


    def arc_segment_to_bezier(cx, cy, rx, ry, phi, theta1, theta2):
        from math import sin, cos, tan, sqrt
        import numpy as np

        cos_phi = cos(phi)
        sin_phi = sin(phi)

        def point(t):
            x = rx * cos(t)
            y = ry * sin(t)
            xp = cos_phi * x - sin_phi * y + cx
            yp = sin_phi * x + cos_phi * y + cy
            return xp, yp

        alpha = sin((theta2 - theta1) / 2.0)
        sin_delta = sin(theta2 - theta1)
        factor = (4.0 / 3.0) * alpha / (1.0 + cos((theta2 - theta1) / 2.0))

        p0 = point(theta1)
        p3 = point(theta2)
        p1 = (
            p0[0] + factor * (-rx * sin(theta1)),
            p0[1] + factor * (ry * cos(theta1))
        )
        p2 = (
            p3[0] + factor * (rx * sin(theta2)),
            p3[1] + factor * (-ry * cos(theta2))
        )

        return [p0, p1, p2, p3]



def arc_to_bezier(x1, y1, rx, ry, phi, large_arc_flag, sweep_flag, x2, y2, tol=1e-9):
    from math import cos, sin, tan, sqrt, radians, ceil, pi, atan2, acos
    import numpy as np


    if not all(isinstance(param, (float, int, np.float32)) for param in [x1, y1, rx, ry, phi, large_arc_flag, sweep_flag, x2, y2]):
        raise TypeError("All parameters to arc_to_bezier must be numeric types.")
    
    # 将角度转换为弧度
    phi_rad = radians(phi)
    cos_phi = cos(phi_rad)
    sin_phi = sin(phi_rad)
    # 计算参数
    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx_sq = rx * rx
    ry_sq = ry * ry
    x1p_sq = x1p * x1p
    y1p_sq = y1p * y1p
    
    # Correct radii
    lambda_ = x1p_sq / rx_sq + y1p_sq / ry_sq
    if lambda_ > 1:
        factor = sqrt(lambda_)
        rx *= factor
        ry *= factor
        rx_sq = rx * rx
        ry_sq = ry * ry
        x1p_sq = x1p * x1p
        y1p_sq = y1p * y1p
    
    radicant = max(0, (rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq) / (rx_sq * y1p_sq + ry_sq * x1p_sq))
    factor = sqrt(radicant) * (1 if large_arc_flag != sweep_flag else -1)
    cxp = factor * (rx * y1p) / ry
    cyp = factor * (-ry * x1p) / rx
    
    # Step 3: Compute (cx, cy) from (cx', cy')
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0
    
    # Step 4: Compute the start and end angles
    def angle(u, v):
        dot = u[0]*v[0] + u[1]*v[1]
        len_u = sqrt(u[0]**2 + u[1]**2)
        len_v = sqrt(v[0]**2 + v[1]**2)
        sign = 1 if (u[0]*v[1] - u[1]*v[0]) >= 0 else -1
        angle_val = acos(min(max(dot / (len_u * len_v), -1.0), 1.0)) * sign
        return angle_val

    v1 = [(x1p - cxp) / rx, (y1p - cyp) / ry]
    v2 = [(-x1p - cxp) / rx, (-y1p - cyp) / ry]
    theta1 = atan2(v1[1], v1[0])
    delta_theta = angle(v1, v2)
    
    if sweep_flag == 0 and delta_theta > 0:
        delta_theta -= 2 * pi
    elif sweep_flag == 1 and delta_theta < 0:
        delta_theta += 2 * pi

    # Number of segments
    segments = max(int(ceil(abs(delta_theta) / (pi / 2))), 1)
    delta = delta_theta / segments

    beziers = []
    for i in range(segments):
        t1 = theta1 + i * delta
        t2 = t1 + delta
        alpha = sin(delta) * (sqrt(4 + 3 * tan(0.5 * delta)**2) - 1) / 3
        
        p1 = [cx + rx * cos(t1) * cos_phi - ry * sin(t1) * sin_phi,
            cy + rx * cos(t1) * sin_phi + ry * sin(t1) * cos_phi]
        p2 = [cx + rx * cos(t2) * cos_phi - ry * sin(t2) * sin_phi,
            cy + rx * cos(t2) * sin_phi + ry * sin(t2) * cos_phi]
        dp1 = [p1[0] - alpha * (rx * sin(t1) * cos_phi + ry * cos(t1) * sin_phi),
            p1[1] - alpha * (rx * sin(t1) * sin_phi - ry * cos(t1) * cos_phi)]
        dp2 = [p2[0] + alpha * (rx * sin(t2) * cos_phi + ry * cos(t2) * sin_phi),
            p2[1] + alpha * (rx * sin(t2) * sin_phi - ry * cos(t2) * cos_phi)]
        
        beziers.append([p1, dp1, dp2, p2])

    return beziers


def quantize_color(color_hex):
    """将 24 位颜色量化为 12 位 (0-4095)"""
    try:
        # 检查颜色字符串是否为标准的 7 位 16 进制颜色代码
        if isinstance(color_hex, str) and color_hex.startswith('#') and len(color_hex) == 7:
            r = int(color_hex[1:3], 16) >> 4  # 将每个颜色通道量化到 4 位 (0-15)
            g = int(color_hex[3:5], 16) >> 4
            b = int(color_hex[5:7], 16) >> 4
            # 合并 RGB 通道（每个通道 4 位，共 12 位）
            return (r << 8) + (g << 4) + b  # 返回一个 12 位的整数 (0-4095)
        else:
                # 对于非标准格式的颜色，返回默认值
            return 0
    except (ValueError, TypeError):
        # 捕获解析错误，返回默认值
        print(f"Warning: Unable to parse color '{color_hex}'. Using default color token.")
        return 0
