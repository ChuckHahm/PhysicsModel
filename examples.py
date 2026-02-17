"""
Example demonstrations of the physics model
"""
from physics_model import (
    Vector, Particle, Gravity, Drag, Spring,
    GravitationalAttraction, PhysicsSimulation
)


def example_projectile():
    """Example: Projectile motion with gravity"""
    print("=" * 50)
    print("Example 1: Projectile Motion")
    print("=" * 50)
    
    # Create a particle at (0, 0) with initial velocity (10, 10)
    projectile = Particle(
        mass=1.0,
        position=Vector(0, 0, 0),
        velocity=Vector(10, 10, 0)
    )
    
    # Create simulation
    sim = PhysicsSimulation(dt=0.01)
    sim.add_particle(projectile)
    sim.add_force(Gravity(g=9.81))
    
    # Run simulation and track trajectory
    print(f"Initial: pos={projectile.position}, vel={projectile.velocity}")
    
    for i in range(0, 200, 20):
        sim.step()
    
    print(f"After 2s: pos={projectile.position}, vel={projectile.velocity}")
    print(f"Max height reached: ~5.1m")
    print()


def example_free_fall():
    """Example: Free fall from height"""
    print("=" * 50)
    print("Example 2: Free Fall")
    print("=" * 50)
    
    # Drop a ball from 100m
    ball = Particle(
        mass=1.0,
        position=Vector(0, 100, 0),
        velocity=Vector(0, 0, 0)
    )
    
    sim = PhysicsSimulation(dt=0.01)
    sim.add_particle(ball)
    sim.add_force(Gravity(g=9.81))
    
    print(f"Initial: height={ball.position.y:.2f}m, velocity={ball.velocity.y:.2f}m/s")
    
    # Simulate until ball hits ground
    while ball.position.y > 0:
        sim.step()
    
    print(f"Final: height={ball.position.y:.2f}m, velocity={ball.velocity.y:.2f}m/s")
    print(f"Time to fall: {sim.time:.2f}s")
    print(f"Impact velocity: ~44.3 m/s (theoretical: sqrt(2*g*h) = {(2*9.81*100)**0.5:.2f})")
    print()


def example_spring_oscillator():
    """Example: Spring oscillator"""
    print("=" * 50)
    print("Example 3: Spring Oscillator")
    print("=" * 50)
    
    # Two particles connected by a spring
    p1 = Particle(mass=1.0, position=Vector(0, 0, 0), velocity=Vector(0, 0, 0))
    p2 = Particle(mass=1.0, position=Vector(3, 0, 0), velocity=Vector(0, 0, 0))
    
    sim = PhysicsSimulation(dt=0.001)
    sim.add_particle(p1)
    sim.add_particle(p2)
    
    # Spring with rest length 1m and stiffness 10 N/m
    spring = Spring(p1, p2, k=10.0, rest_length=1.0)
    sim.add_force(spring)
    
    print(f"Initial distance: {(p2.position - p1.position).magnitude():.2f}m")
    print(f"Rest length: 1.0m")
    
    # Simulate oscillation
    max_dist = 0
    min_dist = float('inf')
    
    for _ in range(5000):
        sim.step()
        dist = (p2.position - p1.position).magnitude()
        max_dist = max(max_dist, dist)
        min_dist = min(min_dist, dist)
    
    print(f"Oscillation range: {min_dist:.2f}m to {max_dist:.2f}m")
    print(f"Final distance: {(p2.position - p1.position).magnitude():.2f}m")
    print()


def example_drag_force():
    """Example: Projectile with air drag"""
    print("=" * 50)
    print("Example 4: Projectile with Air Drag")
    print("=" * 50)
    
    # Two projectiles: one with drag, one without
    projectile_no_drag = Particle(
        mass=1.0,
        position=Vector(0, 0, 0),
        velocity=Vector(20, 20, 0)
    )
    
    projectile_with_drag = Particle(
        mass=1.0,
        position=Vector(0, 0, 0),
        velocity=Vector(20, 20, 0)
    )
    
    # Simulation without drag
    sim1 = PhysicsSimulation(dt=0.01)
    sim1.add_particle(projectile_no_drag)
    sim1.add_force(Gravity(g=9.81))
    
    # Simulation with drag
    sim2 = PhysicsSimulation(dt=0.01)
    sim2.add_particle(projectile_with_drag)
    sim2.add_force(Gravity(g=9.81))
    sim2.add_force(Drag(coefficient=0.1))
    
    # Run both simulations
    for _ in range(200):
        sim1.step()
        sim2.step()
    
    print(f"Without drag: pos={projectile_no_drag.position}")
    print(f"With drag:    pos={projectile_with_drag.position}")
    print(f"Drag reduces range by ~{(1 - projectile_with_drag.position.x/projectile_no_drag.position.x)*100:.1f}%")
    print()


def example_orbital_motion():
    """Example: Two-body orbital system"""
    print("=" * 50)
    print("Example 5: Orbital Motion (Two Bodies)")
    print("=" * 50)
    
    # Large central mass
    sun = Particle(
        mass=1e20,
        position=Vector(0, 0, 0),
        velocity=Vector(0, 0, 0)
    )
    
    # Smaller orbiting mass
    planet = Particle(
        mass=1e10,
        position=Vector(1e8, 0, 0),
        velocity=Vector(0, 8, 0)
    )
    
    sim = PhysicsSimulation(dt=1000)
    sim.add_particle(sun)
    sim.add_particle(planet)
    sim.add_force(GravitationalAttraction(G=6.674e-11))
    
    print(f"Initial distance: {(planet.position - sun.position).magnitude():.2e}m")
    
    # Track orbital motion
    for _ in range(1000):
        sim.step()
    
    print(f"After orbit: distance: {(planet.position - sun.position).magnitude():.2e}m")
    print(f"Position: {planet.position}")
    print()


def example_conservation_laws():
    """Example: Conservation of momentum and energy"""
    print("=" * 50)
    print("Example 6: Conservation Laws")
    print("=" * 50)
    
    # Two particles moving towards each other (collision simulation)
    p1 = Particle(mass=2.0, position=Vector(-5, 0, 0), velocity=Vector(2, 0, 0))
    p2 = Particle(mass=1.0, position=Vector(5, 0, 0), velocity=Vector(-1, 0, 0))
    
    sim = PhysicsSimulation(dt=0.01)
    sim.add_particle(p1)
    sim.add_particle(p2)
    
    initial_momentum = sim.total_momentum()
    initial_energy = sim.total_energy()
    
    print(f"Initial momentum: {initial_momentum}")
    print(f"Initial energy: {initial_energy:.2f}J")
    
    # Simulate motion
    sim.run(100)
    
    final_momentum = sim.total_momentum()
    final_energy = sim.total_energy()
    
    print(f"Final momentum: {final_momentum}")
    print(f"Final energy: {final_energy:.2f}J")
    print(f"Momentum conserved: {abs(initial_momentum.x - final_momentum.x) < 0.01}")
    print(f"Energy conserved: {abs(initial_energy - final_energy) < 0.01}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PHYSICS MODEL DEMONSTRATIONS")
    print("=" * 50 + "\n")
    
    example_projectile()
    example_free_fall()
    example_spring_oscillator()
    example_drag_force()
    example_orbital_motion()
    example_conservation_laws()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
