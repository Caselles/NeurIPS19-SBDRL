
# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2012 Victor Blomqvist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""This submodule contains helper functions to help with quick prototyping
using pymunk together with pygame.
Intended to help with debugging and prototyping, not for actual production use
in a full application. The methods contained in this module is opinionated 
about your coordinate system and not in any way optimized. 
"""

__version__ = "$Id$"
__docformat__ = "reStructuredText"

__all__ = ["to_pygame", "from_pygame"]

flip_y = True


def to_pygame(p, surface):
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates
    """
    if flip_y:
        return int(p[0]), surface.get_height() - int(p[1])
    else:
        return int(p[0]), int(p[1])


def from_pygame(p, surface):
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)
