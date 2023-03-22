import numpy as np


class SegmentBuilder:

    # ---------------------------------------------------------------
    #                       Init Function
    # ---------------------------------------------------------------
    def __init__(self):
        self.length = 2.5
        self.diameter = 0.5
        
        self._segment_handlers = {
            1: self._build_segment_1,
            2: self._build_segment_2,
            3: self._build_segment_3,
            4: self._build_segment_4
        }

    # ---------------------------------------------------------------
    #                       Buid Function
    # ---------------------------------------------------------------
    def build(self, seg_type, x, y, bubble_radius, bubble_lvl):
        """
        Generate bubbles for a given segment type 
        with defined radius and contamination level.

        Parameters
        ----------
        seg_type : int, 1 to 4
            Type of pipe segment
        x : float
            Lower left x coordinate
        y : float
            Lower left y coordinate
        bubble_radius : float
            Bubble radius
        bubble_lvl : float, 0.0 to 0.5
            Contamination percentage of segment
        """
        bubble_lvl = max(min(bubble_lvl, 0.5), 0.0)

        return self._segment_handlers[seg_type](x, y, bubble_radius, bubble_lvl)

    # ---------------------------------------------------------------
    #                Buid Type `1` Segment Function
    # ---------------------------------------------------------------
    def _build_segment_1(self, x, y, bubble_radius, bubble_lvl):
        r"""
        Build bubbles for segment `Type 1`.

        *-------------------------*
        |                         |
        *-------------------------*

        Returns
        -------
        bubble_centres : numpy.ndarray
            List of bubble centers inside segment
        """

        segment_sq = self.diameter * self.length
        bubble_sq = np.pi * bubble_radius ** 2
        bubble_num = int(bubble_lvl * segment_sq / bubble_sq)

        bubble_centres = []
        for i in range(bubble_num):
            rx = np.round(np.random.uniform(x + bubble_radius,
                                            x + self.length - bubble_radius), 4)
            ry = np.round(np.random.uniform(y + bubble_radius,
                                            y + self.diameter - bubble_radius), 4)

            if len(bubble_centres) > 0:
                intersected = False
                for ox, oy in bubble_centres:
                    if np.sqrt((rx - ox)**2 + (ry - oy)**2) < 2 * bubble_radius:
                        intersected = True
                        break
                
                if not intersected:
                    bubble_centres.append((rx, ry))
            else:
                bubble_centres.append((rx, ry))

        return np.array(bubble_centres)

    # ---------------------------------------------------------------
    #                Buid Type `2` Segment Function
    # ---------------------------------------------------------------
    def _build_segment_2(self, x, y, bubble_radius, bubble_lvl):
        r"""
        Build bubbles for segment `Type 2`.

        *----*
        |    |
        |    |
        |    |
        |    |
        |    |
        |    |
        |    |
        *----*

        Returns
        -------
        bubble_centres : np.ndarray
            List of bubble centers inside segment
        """

        bubble_centres = self._build_segment_1(x, y, bubble_radius, bubble_lvl)

        theta = np.deg2rad(90)
        rot = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        
        bubble_centres = np.array([[x, y]]) + \
                         np.apply_along_axis(rot.dot, 1, bubble_centres - np.array([[x, y]])) + \
                         np.array([[self.diameter, 0.0]])

        return bubble_centres

    # ---------------------------------------------------------------
    #                Buid Type `3` Segment Function
    # ---------------------------------------------------------------
    def _build_segment_3(self, x, y, bubble_radius, bubble_lvl):
        r"""
        Build bubbles for segment `Type 3`.

          /* -45 deg
         /  \
        *    \
        \     \
         \     \
          \     \
           \     *
            \   / 
             \ /
              *

        Returns
        -------
        bubble_centres : np.ndarray
            List of bubble centers inside segment
        """

        bubble_centres = self._build_segment_1(x, y, bubble_radius, bubble_lvl)

        theta = np.deg2rad(-45)
        rot = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        
        bubble_centres = np.array([[x, y]]) + \
                         np.apply_along_axis(rot.dot, 1, bubble_centres - np.array([[x, y]])) + \
                         np.array([[self.diameter, 0.0]])

        return bubble_centres

    # ---------------------------------------------------------------
    #                Buid Type `4` Segment Function
    # ---------------------------------------------------------------
    def _build_segment_4(self, x, y, bubble_radius, bubble_lvl):
        r"""
        Build bubbles for segment `Type 4`.

                            * 45 deg
                           / \
        *----------------*/    \
        |                       /
        *----------------*-----*

        Returns
        -------
        bubble_centres : np.ndarray
            List of bubble centers inside segment
        """

        theta = np.deg2rad(45)
        rot = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
        rot_point = np.round(np.array([[x + self.length - np.pi/4, 
                                        y + 2 * self.diameter]]), 4)
        
        # Segment area = area of rectangle + area of sector
        segment_sq = self.diameter * rot_point[0][0] + \
                     theta / 2 * 3 * self.diameter**2
        print(f'Area: {segment_sq}, rot_point: {rot_point}')
        bubble_sq = np.pi * bubble_radius ** 2
        bubble_num = int(bubble_lvl * segment_sq / bubble_sq)

        bubble_centres = []
        for i in range(bubble_num):
            rx = np.round(np.random.uniform(x + bubble_radius,
                                            x + self.length - bubble_radius), 4)
            ry = np.round(np.random.uniform(y + bubble_radius,
                                            y + self.diameter - bubble_radius), 4)
            # TODO: bending
            # if rx > rot_point[0][0]:
            #     print(f"rx, ry: {np.array([rx, ry])}")
            #     rx, ry = rot_point[0] + np.dot(rot, np.array([rx, ry]) - rot_point[0])

            if len(bubble_centres) > 0:
                intersected = False
                for ox, oy in bubble_centres:
                    if np.sqrt((rx - ox)**2 + (ry - oy)**2) <= self.diameter:
                        intersected = True
                
                if not intersected:
                    bubble_centres.append((rx, ry))
            else:
                bubble_centres.append((rx, ry))

        return np.array(bubble_centres)
