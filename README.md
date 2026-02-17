# PhysicsModel

A comprehensive Python-based physics simulation framework for modeling particle dynamics, forces, and physical interactions.

## Features

- **Vector Mathematics**: Full 3D vector operations (addition, subtraction, multiplication, normalization, dot product)
- **Particle System**: Mass-based particles with position, velocity, and acceleration
- **Force Models**:
  - Gravity (constant gravitational field)
  - Drag (air resistance)
  - Spring forces (Hooke's law)
  - Gravitational attraction (Newton's law of universal gravitation)
- **Physics Simulation Engine**: Time-stepped integration with Euler method
- **Conservation Laws**: Support for energy and momentum calculations

## Installation

No external dependencies required! Just Python 3.6+

```bash
git clone https://github.com/ChuckHahm/PhysicsModel.git
cd PhysicsModel
```

## Quick Start

### Basic Example: Free Fall

```python
from physics_model import Vector, Particle, Gravity, PhysicsSimulation

# Create a particle at height 10m
ball = Particle(mass=1.0, position=Vector(0, 10, 0), velocity=Vector(0, 0, 0))

# Create simulation with gravity
sim = PhysicsSimulation(dt=0.01)
sim.add_particle(ball)
sim.add_force(Gravity(g=9.81))

# Simulate for 100 steps (1 second)
sim.run(100)

print(f"Position: {ball.position}")
print(f"Velocity: {ball.velocity}")
```

### Projectile Motion

```python
from physics_model import Vector, Particle, Gravity, PhysicsSimulation

# Launch projectile at 45 degrees
projectile = Particle(
    mass=1.0,
    position=Vector(0, 0, 0),
    velocity=Vector(10, 10, 0)
)

sim = PhysicsSimulation(dt=0.01)
sim.add_particle(projectile)
sim.add_force(Gravity(g=9.81))

# Simulate trajectory
sim.run(200)
```

### Spring Oscillation

```python
from physics_model import Vector, Particle, Spring, PhysicsSimulation

# Two particles connected by spring
p1 = Particle(mass=1.0, position=Vector(0, 0, 0))
p2 = Particle(mass=1.0, position=Vector(3, 0, 0))

sim = PhysicsSimulation(dt=0.001)
sim.add_particle(p1)
sim.add_particle(p2)

# Add spring force (k=10 N/m, rest length=1m)
spring = Spring(p1, p2, k=10.0, rest_length=1.0)
sim.add_force(spring)

# Watch the oscillation
sim.run(5000)
```

## Core Classes

### Vector
3D vector class supporting standard operations:
- Addition, subtraction, multiplication, division
- Magnitude and normalization
- Dot product

### Particle
Represents a physical particle with:
- Mass
- Position (Vector)
- Velocity (Vector)
- Methods for kinetic energy and momentum calculation

### Forces
Base class with specific implementations:
- **Gravity**: Constant gravitational field (F = m*g)
- **Drag**: Velocity-dependent air resistance (F = -k*v)
- **Spring**: Hooke's law spring force (F = -k*Δx)
- **GravitationalAttraction**: Newton's universal gravitation (F = G*m1*m2/r²)

### PhysicsSimulation
Main simulation engine that:
- Manages particles and forces
- Steps through time using Euler integration
- Calculates system energy and momentum

## Running Tests

```bash
python test_physics_model.py
```

The test suite includes:
- Vector operations
- Particle dynamics
- Force models
- Energy and momentum conservation
- Integration accuracy

## Running Examples

```bash
python examples.py
```

This demonstrates:
1. Projectile motion
2. Free fall
3. Spring oscillator
4. Air drag effects
5. Orbital motion
6. Conservation laws

## Physics Principles

### Newton's Laws
- **First Law**: Objects maintain velocity unless acted upon by force
- **Second Law**: F = ma (implemented in Particle.update)
- **Third Law**: Action-reaction pairs (in gravitational attraction)

### Energy Conservation
Kinetic energy: KE = ½mv²
(Note: Potential energy not directly calculated but implicit in forces)

### Momentum Conservation
Linear momentum: p = mv
(Conserved in closed systems without external forces)

## Limitations

- Uses Euler integration (first-order accuracy)
- No collision detection/response
- No constraint solving
- No rigid body dynamics
- Numerical errors accumulate over long simulations

## Future Enhancements

- Verlet integration for better energy conservation
- Collision detection and response
- Constraint-based physics
- Rigid body dynamics
- 3D visualization support

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.