import pyglet

window = pyglet.window.Window()

label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

@window.event
def on_draw():
    window.clear()
    label.draw()

@window.event
def on_resize(width, height):
    print(f'The window was resized to {width},{height}')
    
event_logger = pyglet.window.event.WindowEventLogger()
window.push_handlers(event_logger)
