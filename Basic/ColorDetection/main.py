import cv2
import pandas as pd
import argparse

# Arguments via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args["image"]

# Read image with opencv
img = cv2.imread(img_path)

# Read colors csv file
index = ["color", "color_name", "hex", "R", "G", "B"]
color_df = pd.read_csv(".\\colors.csv", names=index, header=None)

clicked = False


# Sets the values of r,g,b and x and y positions when mouse is double clicked
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


# Returns the color of the chosen pixel
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(color_df)):
        d = (
            abs(R - int(color_df.loc[i, "R"]))
            + abs(G - int(color_df.loc[i, "G"]))
            + abs(B - int(color_df.loc[i, "B"]))
        )
        if d <= minimum:
            minimum = d
            cname = color_df.loc[i, "color_name"]
    return cname


cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_function)

while True:
    cv2.imshow("image", img)
    if clicked:
        # Create rectangle on image with specified r, g, b values
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Create text string to display r, g, b values
        text = (
            get_color_name(r, g, b) + " R=" + str(r) + " G=" + str(g) + " B=" + str(b)
        )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # If color is light, switch text to black
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Close if esc is pressed
    if cv2.waitKey(20) & 0xFF == 27:
        break

    # Close if "X" on window is clicked
    if cv2.getWindowProperty("image", 1) < 0:
        break

cv2.destroyAllWindows()
