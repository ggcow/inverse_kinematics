import pygame
from math import sin, cos
import autograd.numpy as np
from autograd import jacobian
from itertools import accumulate


def cost(angles):
    end_x = LINK_LENGTHS.dot(np.cos(angles))
    end_y = LINK_LENGTHS.dot(np.sin(angles))
    return np.array([end_x, end_y])
jacobian_cost = jacobian(cost)


SCREEN_DIMENSIONS = np.array([800, 600])

N = 1
LINK_ANGLES = np.zeros(N)
LINK_LENGTHS = np.ones(N) * 100

# 60 FPS
FPS = 60
clock = pygame.time.Clock()


pygame.init()
screen = pygame.display.set_mode(SCREEN_DIMENSIONS, pygame.RESIZABLE)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        if event.type == pygame.VIDEORESIZE:
            SCREEN_DIMENSIONS = np.array([event.w, event.h])
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                LINK_LENGTHS *= 1.1
            if event.button == 5:
                LINK_LENGTHS *= 0.9
        if event.type == pygame.KEYDOWN and event.unicode == "+":
            LINK_ANGLES = np.append(LINK_ANGLES, 0)
            LINK_LENGTHS = np.append(LINK_LENGTHS, LINK_LENGTHS[-1])
        if event.type == pygame.KEYDOWN and event.unicode == "-" and len(LINK_ANGLES) > 1:
            LINK_ANGLES = np.delete(LINK_ANGLES, -1)
            LINK_LENGTHS = np.delete(LINK_LENGTHS, -1)

    ## Inverse Kinematics
    mouse_position = np.array(pygame.mouse.get_pos()) - SCREEN_DIMENSIONS / 2
    if (dist := np.linalg.norm(mouse_position)) > sum(LINK_LENGTHS):
        mouse_position *= sum(LINK_LENGTHS) / dist
    LINK_ANGLES += np.cumsum(np.linalg.pinv(jacobian_cost(LINK_ANGLES)).dot(mouse_position - cost(LINK_ANGLES))) * 0.1
        
    ## Graphics
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (100, 100, 0), (mouse_position + SCREEN_DIMENSIONS / 2), 10)
    points = list(accumulate(zip(LINK_ANGLES, LINK_LENGTHS), lambda prev, val: prev + np.array([cos(val[0]) * val[1], sin(val[0]) * val[1]]), initial=SCREEN_DIMENSIONS / 2))
    pygame.draw.lines(screen, (255, 255, 255), False, points, 5)
    pygame.display.update()

    clock.tick(FPS)
pygame.quit()

