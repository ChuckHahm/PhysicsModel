"""
Physics Model - A simple physics simulation framework
"""
import math
from typing import List, Tuple


class Vector:
    """3D Vector class for physics calculations"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector':
        """Return a normalized (unit) vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0, 0)
        return self / mag
    
    def dot(self, other: 'Vector') -> float:
        """Dot product with another vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def __repr__(self) -> str:
        return f"Vector({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Particle:
    """Represents a particle with mass, position, velocity, and acceleration"""
    
    def __init__(self, mass: float, position: Vector = None, velocity: Vector = None):
        self.mass = mass
        self.position = position if position else Vector()
        self.velocity = velocity if velocity else Vector()
        self.acceleration = Vector()
        self.forces = Vector()
    
    def apply_force(self, force: Vector):
        """Apply a force to the particle"""
        self.forces = self.forces + force
    
    def update(self, dt: float):
        """Update particle state using Euler integration"""
        # F = ma => a = F/m
        self.acceleration = self.forces / self.mass
        
        # Update velocity: v = v + a*dt
        self.velocity = self.velocity + self.acceleration * dt
        
        # Update position: p = p + v*dt
        self.position = self.position + self.velocity * dt
        
        # Reset forces for next iteration
        self.forces = Vector()
    
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy: KE = 0.5 * m * v^2"""
        return 0.5 * self.mass * self.velocity.magnitude()**2
    
    def momentum(self) -> Vector:
        """Calculate momentum: p = m * v"""
        return self.velocity * self.mass
    
    def __repr__(self) -> str:
        return f"Particle(m={self.mass}, pos={self.position}, vel={self.velocity})"


class Force:
    """Base class for forces"""
    
    def apply(self, particle: Particle, particles: List[Particle] = None):
        """Apply force to a particle"""
        raise NotImplementedError


class Gravity(Force):
    """Gravitational force"""
    
    def __init__(self, g: float = 9.81):
        self.g = g
    
    def apply(self, particle: Particle, particles: List[Particle] = None):
        """Apply gravitational force: F = m * g"""
        force = Vector(0, -self.g * particle.mass, 0)
        particle.apply_force(force)


class Drag(Force):
    """Air drag force"""
    
    def __init__(self, coefficient: float = 0.1):
        self.coefficient = coefficient
    
    def apply(self, particle: Particle, particles: List[Particle] = None):
        """Apply drag force: F = -k * v"""
        drag = particle.velocity * (-self.coefficient)
        particle.apply_force(drag)


class Spring(Force):
    """Spring force between two particles"""
    
    def __init__(self, particle_a: Particle, particle_b: Particle, 
                 k: float = 10.0, rest_length: float = 1.0):
        self.particle_a = particle_a
        self.particle_b = particle_b
        self.k = k
        self.rest_length = rest_length
    
    def apply(self, particle: Particle, particles: List[Particle] = None):
        """Apply spring force: F = -k * (x - x0)"""
        if particle == self.particle_a:
            direction = self.particle_b.position - self.particle_a.position
            distance = direction.magnitude()
            if distance > 0:
                force = direction.normalize() * self.k * (distance - self.rest_length)
                particle.apply_force(force)
        elif particle == self.particle_b:
            direction = self.particle_a.position - self.particle_b.position
            distance = direction.magnitude()
            if distance > 0:
                force = direction.normalize() * self.k * (distance - self.rest_length)
                particle.apply_force(force)


class GravitationalAttraction(Force):
    """Gravitational attraction between particles"""
    
    def __init__(self, G: float = 6.674e-11):
        self.G = G
    
    def apply(self, particle: Particle, particles: List[Particle] = None):
        """Apply gravitational attraction: F = G * m1 * m2 / r^2"""
        if particles is None:
            return
        
        for other in particles:
            if other == particle:
                continue
            
            direction = other.position - particle.position
            distance = direction.magnitude()
            
            if distance > 0:
                force_magnitude = self.G * particle.mass * other.mass / (distance ** 2)
                force = direction.normalize() * force_magnitude
                particle.apply_force(force)


class PhysicsSimulation:
    """Physics simulation engine"""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.particles: List[Particle] = []
        self.forces: List[Force] = []
        self.time = 0.0
    
    def add_particle(self, particle: Particle):
        """Add a particle to the simulation"""
        self.particles.append(particle)
    
    def add_force(self, force: Force):
        """Add a force to the simulation"""
        self.forces.append(force)
    
    def step(self):
        """Advance simulation by one time step"""
        # Apply all forces to all particles
        for force in self.forces:
            for particle in self.particles:
                force.apply(particle, self.particles)
        
        # Update all particles
        for particle in self.particles:
            particle.update(self.dt)
        
        self.time += self.dt
    
    def run(self, steps: int):
        """Run simulation for a number of steps"""
        for _ in range(steps):
            self.step()
    
    def total_energy(self) -> float:
        """Calculate total kinetic energy of the system"""
        return sum(p.kinetic_energy() for p in self.particles)
    
    def total_momentum(self) -> Vector:
        """Calculate total momentum of the system"""
        total = Vector()
        for p in self.particles:
            total = total + p.momentum()
        return total
