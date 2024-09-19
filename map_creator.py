import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


# Map Initialization
map_size = 500
free_space_value = 195
obstacle_value = 127
start_position_value = 208
boundary_width = 30
map_array = np.full((map_size, map_size), free_space_value, dtype=int)
map_array[:boundary_width, :] = obstacle_value
map_array[-boundary_width:, :] = obstacle_value
map_array[:, :boundary_width] = obstacle_value
map_array[:, -boundary_width:] = obstacle_value
temp_map_array = np.copy(map_array)

# Create a matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.2, bottom=0.2)
map_display = ax.imshow(map_array, cmap='gray', vmin=0, vmax=255)

# State variables
drawing_mode = None
line_start = None
cursor_square = ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=None, edgecolor='red', visible=False))
reset_state = False

map_file_name = 'map.png'
os.makedirs('./maps_spec', exist_ok=True)


# Function to update the map display
def update_display():
    map_display.set_data(temp_map_array)
    fig.canvas.draw()

# Event handlers
def on_press(event):
    global line_start, drawing_mode
    if event.inaxes != ax or drawing_mode is None:
        return
    line_start = (int(event.ydata), int(event.xdata))


def on_release(event):
    global line_start, drawing_mode, temp_map_array
    if event.inaxes != ax or drawing_mode is None or line_start is None:
        return
    line_end = (int(event.ydata), int(event.xdata))

    dx = abs(line_end[1] - line_start[1])
    dy = abs(line_end[0] - line_start[0])

    if drawing_mode in ['obstacle', 'free_space']:
        if dx > dy:
            # Make the line horizontal
            line_end = (line_start[0], line_end[1])
        else:
            # Make the line vertical
            line_end = (line_end[0], line_start[1])

        draw_line_on_map(line_start, line_end, map_array, get_drawing_value())
    elif drawing_mode == 'start':
        draw_start_position(line_start)

    line_start = None
    temp_map_array = np.copy(map_array)
    update_display()

def on_move(event):
    global temp_map_array, cursor_square
    if event.inaxes == ax:
        cursor_pos.set_text(f'Cursor: ({int(event.xdata)}, {int(event.ydata)})')
        update_cursor_square(event)
        if drawing_mode and line_start is not None:
            line_end = (int(event.ydata), int(event.xdata))
            temp_array = np.copy(map_array)
            draw_line_on_map(line_start, line_end, temp_array, get_drawing_value())
            temp_map_array = temp_array
            update_display()
        fig.canvas.draw_idle()

def update_cursor_square(event):
    if drawing_mode == 'obstacle':
        edge_color = 'red'
    elif drawing_mode == 'free_space':
        edge_color = 'yellow'
    else:
        edge_color = 'black'
    size = get_cursor_square_size()
    cursor_square.set_width(size)
    cursor_square.set_height(size)
    cursor_square.set_xy((event.xdata - size/2 - 1, event.ydata - size/2 - 1))
    cursor_square.set_edgecolor(edge_color)
    cursor_square.set_visible(True)
    fig.canvas.draw_idle()

def get_cursor_square_size():
    if drawing_mode == 'obstacle' or drawing_mode == 'free_space':
        return slider_thickness.val
    elif drawing_mode == 'start':
        return 25
    else:
        return 0

def get_drawing_value():
    if drawing_mode == 'obstacle':
        return obstacle_value
    elif drawing_mode == 'free_space':
        return free_space_value
    else:
        return None

def draw_line_on_map(start, end, array, value):
    if value is None:
        return
    thickness = int(slider_thickness.val) // 2
    y0, x0 = start
    y1, x1 = end
    num = max(abs(x1 - x0), abs(y1 - y0)) + 1
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    for i in range(len(x)):
        xi, yi = int(x[i]), int(y[i])
        array[max(yi-thickness, 0):min(yi+thickness, map_size), max(xi-thickness, 0):min(xi+thickness, map_size)] = value

def draw_start_position(position):
    y, x = position
    map_array[max(y-12, 0):min(y+13, map_size), max(x-12, 0):min(x+13, map_size)] = start_position_value

def toggle_drawing_mode(mode):
    global drawing_mode
    drawing_mode = mode

def reset_map(_):
    global reset_state, map_array, temp_map_array
    if reset_state:
        map_array.fill(free_space_value)
    else:
        map_array.fill(obstacle_value)
    temp_map_array = np.copy(map_array)
    reset_state = not reset_state
    update_display()

def save_map(_):
    cv2.imwrite(f'maps_spec/{map_file_name}', map_array)

# Add buttons and slider
ax_button_obstacle = plt.axes([0.1, 0.05, 0.15, 0.075])
button_obstacle = Button(ax_button_obstacle, 'Obstacle')
button_obstacle.on_clicked(lambda _: toggle_drawing_mode('obstacle'))

ax_button_free_space = plt.axes([0.25, 0.05, 0.2, 0.075])
button_free_space = Button(ax_button_free_space, 'Free Space')
button_free_space.on_clicked(lambda _: toggle_drawing_mode('free_space'))

ax_button_start = plt.axes([0.45, 0.05, 0.15, 0.075])
button_start = Button(ax_button_start, 'Start')
button_start.on_clicked(lambda _: toggle_drawing_mode('start'))

ax_button_reset = plt.axes([0.6, 0.05, 0.15, 0.075])
button_reset = Button(ax_button_reset, 'Reset')
button_reset.on_clicked(reset_map)

ax_button_save = plt.axes([0.75, 0.05, 0.15, 0.075])
button_save = Button(ax_button_save, 'Save Map')
button_save.on_clicked(save_map)

ax_slider_thickness = plt.axes([0.1, 0.15, 0.8, 0.02])
slider_thickness = Slider(ax_slider_thickness, 'Thickness', 5, 120, valinit=30, valstep=1)

cursor_pos = fig.text(0.1, 0.9, 'Cursor: (0, 0)', fontsize=10)

# Connect the events
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

# Show the plot
plt.show()
