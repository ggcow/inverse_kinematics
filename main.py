import pygame
from math import sin, cos, atan2
import numpy


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

SCREEN_DIMENSIONS = numpy.array([SCREEN_WIDTH, SCREEN_HEIGHT])

LINK_ANGLES = numpy.array([0.1, 0.1])
LINK_LENGTHS = [150, 100]


def inverse_kinematics(joint_angles, link_lengths, target_position):
    for _ in range(10):
        a1, a2 = joint_angles
        l1, l2 = link_lengths
        inverted_jacobian = numpy.array([
            [l2 * cos(a1 + a2), l1 * sin(a1 + a2)],
            [-l1 * cos(a1) - l2 * cos(a1 + a2), -l1 * sin(a1) - l2 * sin(a1 + a2)],
        ]) / (sin(a2) * l1 * l2 or 1)
        end_effector = numpy.array([l2 * cos(a1+a2) + l1 * cos(a1), l2 * sin(a1+a2) + l1 * sin(a1)])
        joint_angles += inverted_jacobian.dot(target_position - end_effector)
    return joint_angles

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    mouse_position = numpy.array(pygame.mouse.get_pos()) - SCREEN_DIMENSIONS / 2

    if numpy.linalg.norm(mouse_position) > sum(LINK_LENGTHS)**2:
        LINK_ANGLES = numpy.array([atan2(mouse_position[1], mouse_position[0]), 0])
    else:
        LINK_ANGLES = inverse_kinematics(LINK_ANGLES, LINK_LENGTHS, mouse_position)
        

    x, y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    total_angle = 0
    screen.fill((0, 0, 0))
    for angle, length in zip(LINK_ANGLES, LINK_LENGTHS):
        total_angle += angle
        pygame.draw.circle(screen, (255,) * 3, (x, y), 10)
        x2 = x + cos(total_angle) * length
        y2 = y + sin(total_angle) * length
        pygame.draw.line(screen, (255,) * 3, (x, y), (x2, y2), 5)
        x, y = x2, y2
    pygame.draw.circle(screen, (255,) * 3, (x, y), 10)
    pygame.display.update()
pygame.quit()
