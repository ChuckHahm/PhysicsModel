#!/usr/bin/env python3
"""
Simple ASCII visualization of a bouncing ball
"""
from physics_model import Vector, Particle, Gravity, PhysicsSimulation
import time
import os


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def draw_ball(height, max_height=20, width=40):
    """Draw the ball at a given height"""
    lines = []
    height_pos = int((height / max_height) * 20)
    height_pos = max(0, min(19, height_pos))
    
    for i in range(19, -1, -1):
        if i == height_pos:
            line = " " * (width // 2) + "O"
        else:
            line = ""
        lines.append(line)
    
    lines.append("-" * width)
    lines.append(f"Height: {height:.2f}m")
    return "\n".join(lines)


def bouncing_ball_demo():
    """Demonstrate a bouncing ball with visualization"""
    print("Bouncing Ball Simulation")
    print("Press Ctrl+C to stop")
    print()
    time.sleep(2)
    
    # Create ball at 20m height
    ball = Particle(mass=1.0, position=Vector(0, 20, 0), velocity=Vector(0, 0, 0))
    
    # Create simulation
    sim = PhysicsSimulation(dt=0.01)
    sim.add_particle(ball)
    sim.add_force(Gravity(g=9.81))
    
    try:
        while True:
            # Update simulation
            sim.step()
            
            # Simple bounce at ground
            if ball.position.y <= 0:
                ball.position.y = 0
                ball.velocity.y = -ball.velocity.y * 0.9  # 90% energy retained
            
            # Draw every 5 frames (0.05s)
            if int(sim.time * 100) % 5 == 0:
                clear_screen()
                print(draw_ball(ball.position.y))
                print(f"Velocity: {ball.velocity.y:.2f} m/s")
                print(f"Time: {sim.time:.2f}s")
                time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nSimulation ended")


if __name__ == "__main__":
    bouncing_ball_demo()
