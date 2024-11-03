import numpy as np
from OpenGL import GL as gl

class TexturePool:
    """Manages a pool of reusable OpenGL textures to avoid constant creation/deletion"""

    def __init__(self):
        self.available_textures: list[int] = []  # Pool of unused texture IDs
        self.in_use_textures: list[int] = []  # Currently allocated texture IDs

    def rent(self) -> int:
        """Get a texture ID from the pool or create new if needed"""
        if self.available_textures:
            tex_id = self.available_textures.pop()
        else:
            tex_id = self._create()
        self.in_use_textures.append(tex_id)
        return tex_id

    def release(self, tex_id: int):
        """Return a texture ID to the available pool"""
        if tex_id in self.in_use_textures:
            self.in_use_textures.remove(tex_id)
            self.available_textures.append(tex_id)

    def _create(self) -> int:
        """Create a new OpenGL texture with standard parameters"""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        return texture_id

    def update_texture(self, texture_id: int, image: np.ndarray):
        """Update texture data for given ID"""
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        # Ensure RGB format and contiguous C-order array
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        # Convert float32 [0,1] to uint8 [0,255]
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB8,
            image.shape[1], image.shape[0], 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
            image.tobytes()
        )
