import pygame
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random  # Add this import for epsilon-greedy exploration

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Platformer with PyTorch AI (Debugged)")

WHITE, BLACK, RED, GREEN, BLUE, YELLOW = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)

PLAYER_WIDTH, PLAYER_HEIGHT = 40, 60
player_x, player_y = 50, HEIGHT - PLAYER_HEIGHT - 10
player_speed = 5
jump_speed = -20
gravity = 0.6

platforms = [
    pygame.Rect(0, HEIGHT - 10, WIDTH, 10),
    pygame.Rect(300, 450, 200, 20),
    pygame.Rect(100, 350, 200, 20),
    pygame.Rect(500, 250, 200, 20),
]

goal = pygame.Rect(WIDTH - 50, 50, 30, 30)

player_vel_y = 0
is_jumping = False
score = 0

clock = pygame.time.Clock()

ai_player = pygame.Rect(50, HEIGHT - PLAYER_HEIGHT - 10, PLAYER_WIDTH, PLAYER_HEIGHT)
ai_vel_y = 0
ai_is_jumping = False

class AINet(nn.Module):
    def __init__(self):
        super(AINet, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Remove softmax to get raw Q-values

ai_net = AINet()
optimizer = optim.Adam(ai_net.parameters(), lr=0.001)

def get_state():
    return torch.tensor([
        ai_player.x / WIDTH,
        ai_player.y / HEIGHT,
        ai_vel_y / 20,
        goal.x / WIDTH,
        goal.y / HEIGHT,
        min([p.bottom for p in platforms if p.bottom > ai_player.bottom] + [HEIGHT]) / HEIGHT
    ], dtype=torch.float32).unsqueeze(0)

epsilon = 0.3  # Exploration rate

def ai_move():
    global ai_vel_y, ai_is_jumping
    
    state = get_state()
    with torch.no_grad():
        q_values = ai_net(state)
    
    # Epsilon-greedy action selection
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        action = torch.argmax(q_values).item()
    
    # Debug print
    print(f"AI Action: {action}, Q-values: {q_values.numpy()}")
    
    if action == 0 and ai_player.left > 0:  # Move left
        ai_player.x -= player_speed
    elif action == 1 and ai_player.right < WIDTH:  # Move right
        ai_player.x += player_speed
    elif action == 2 and not ai_is_jumping:  # Jump
        ai_vel_y = jump_speed
        ai_is_jumping = True
    
    ai_vel_y, ai_is_jumping = apply_gravity(ai_player, ai_vel_y, ai_is_jumping)

def handle_movement(keys, player_rect):
    global player_vel_y, is_jumping
    
    if keys[pygame.K_LEFT] and player_rect.left > 0:
        player_rect.x -= player_speed
    if keys[pygame.K_RIGHT] and player_rect.right < WIDTH:
        player_rect.x += player_speed
    
    if keys[pygame.K_SPACE] and not is_jumping:
        player_vel_y = jump_speed
        is_jumping = True

def apply_gravity(player_rect, vel_y, is_jumping):
    vel_y += gravity
    player_rect.y += vel_y
    
    for platform in platforms:
        if player_rect.colliderect(platform):
            if vel_y > 0:
                player_rect.bottom = platform.top
                is_jumping = False
                vel_y = 0
            elif vel_y < 0:
                player_rect.top = platform.bottom
                vel_y = 0
    
    return vel_y, is_jumping

def check_goal(player_rect, goal_rect):
    global score
    if player_rect.colliderect(goal_rect):
        score += 1
        print(f"Level complete! Score: {score}")
        return True
    return False

def game_loop():
    global player_vel_y, is_jumping, score
    
    player_rect = pygame.Rect(player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        keys = pygame.key.get_pressed()
        handle_movement(keys, player_rect)
        player_vel_y, is_jumping = apply_gravity(player_rect, player_vel_y, is_jumping)
        
        ai_move()
        
        if check_goal(player_rect, goal):
            player_rect.x, player_rect.y = 50, HEIGHT - PLAYER_HEIGHT - 10
        
        if check_goal(ai_player, goal):
            ai_player.x, ai_player.y = 50, HEIGHT - PLAYER_HEIGHT - 10
        
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLUE, player_rect)
        pygame.draw.rect(screen, YELLOW, ai_player)
        for platform in platforms:
            pygame.draw.rect(screen, GREEN, platform)
        pygame.draw.rect(screen, RED, goal)
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    game_loop()