"""
Tests for the physics model
"""
import unittest
import math
from physics_model import (
    Vector, Particle, Gravity, Drag, Spring, 
    GravitationalAttraction, PhysicsSimulation
)


class TestVector(unittest.TestCase):
    """Test Vector class"""
    
    def test_init(self):
        v = Vector(1, 2, 3)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)
    
    def test_addition(self):
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        v3 = v1 + v2
        self.assertEqual(v3.x, 5)
        self.assertEqual(v3.y, 7)
        self.assertEqual(v3.z, 9)
    
    def test_subtraction(self):
        v1 = Vector(4, 5, 6)
        v2 = Vector(1, 2, 3)
        v3 = v1 - v2
        self.assertEqual(v3.x, 3)
        self.assertEqual(v3.y, 3)
        self.assertEqual(v3.z, 3)
    
    def test_multiplication(self):
        v = Vector(1, 2, 3)
        v2 = v * 2
        self.assertEqual(v2.x, 2)
        self.assertEqual(v2.y, 4)
        self.assertEqual(v2.z, 6)
    
    def test_division(self):
        v = Vector(2, 4, 6)
        v2 = v / 2
        self.assertEqual(v2.x, 1)
        self.assertEqual(v2.y, 2)
        self.assertEqual(v2.z, 3)
    
    def test_magnitude(self):
        v = Vector(3, 4, 0)
        self.assertEqual(v.magnitude(), 5)
    
    def test_normalize(self):
        v = Vector(3, 4, 0)
        v_norm = v.normalize()
        self.assertAlmostEqual(v_norm.magnitude(), 1.0)
        self.assertAlmostEqual(v_norm.x, 0.6)
        self.assertAlmostEqual(v_norm.y, 0.8)
    
    def test_dot_product(self):
        v1 = Vector(1, 2, 3)
        v2 = Vector(4, 5, 6)
        self.assertEqual(v1.dot(v2), 32)


class TestParticle(unittest.TestCase):
    """Test Particle class"""
    
    def test_init(self):
        p = Particle(mass=1.0, position=Vector(0, 0, 0))
        self.assertEqual(p.mass, 1.0)
        self.assertEqual(p.position.x, 0)
    
    def test_apply_force(self):
        p = Particle(mass=1.0)
        force = Vector(1, 0, 0)
        p.apply_force(force)
        self.assertEqual(p.forces.x, 1)
    
    def test_update(self):
        p = Particle(mass=1.0, position=Vector(0, 0, 0), velocity=Vector(0, 0, 0))
        force = Vector(1, 0, 0)  # 1 N force
        p.apply_force(force)
        p.update(dt=1.0)  # 1 second
        
        # With F=1N and m=1kg, a=1m/s^2
        # After 1s: v=1m/s, position=1m
        self.assertAlmostEqual(p.velocity.x, 1.0)
        self.assertAlmostEqual(p.position.x, 1.0)
    
    def test_kinetic_energy(self):
        p = Particle(mass=2.0, velocity=Vector(3, 0, 0))
        # KE = 0.5 * m * v^2 = 0.5 * 2 * 9 = 9
        self.assertAlmostEqual(p.kinetic_energy(), 9.0)
    
    def test_momentum(self):
        p = Particle(mass=2.0, velocity=Vector(3, 0, 0))
        momentum = p.momentum()
        self.assertAlmostEqual(momentum.x, 6.0)


class TestGravity(unittest.TestCase):
    """Test Gravity force"""
    
    def test_gravity_force(self):
        p = Particle(mass=1.0, position=Vector(0, 10, 0))
        gravity = Gravity(g=9.81)
        gravity.apply(p)
        
        # Force should be -9.81 in y direction
        self.assertAlmostEqual(p.forces.y, -9.81)
    
    def test_free_fall(self):
        p = Particle(mass=1.0, position=Vector(0, 10, 0), velocity=Vector(0, 0, 0))
        sim = PhysicsSimulation(dt=0.01)
        sim.add_particle(p)
        sim.add_force(Gravity(g=9.81))
        
        # Simulate for 1 second
        sim.run(100)
        
        # After 1s of free fall: v = g*t = 9.81 m/s
        # position = h0 - 0.5*g*t^2 = 10 - 0.5*9.81*1 = 5.095
        self.assertAlmostEqual(p.velocity.y, -9.81, delta=0.1)
        self.assertAlmostEqual(p.position.y, 5.095, delta=0.1)


class TestDrag(unittest.TestCase):
    """Test Drag force"""
    
    def test_drag_force(self):
        p = Particle(mass=1.0, velocity=Vector(10, 0, 0))
        drag = Drag(coefficient=0.1)
        drag.apply(p)
        
        # Drag should oppose velocity
        self.assertAlmostEqual(p.forces.x, -1.0)


class TestSpring(unittest.TestCase):
    """Test Spring force"""
    
    def test_spring_force(self):
        p1 = Particle(mass=1.0, position=Vector(0, 0, 0))
        p2 = Particle(mass=1.0, position=Vector(2, 0, 0))
        spring = Spring(p1, p2, k=10.0, rest_length=1.0)
        
        spring.apply(p1)
        
        # Distance is 2, rest length is 1, so extension is 1
        # Force = k * extension = 10 * 1 = 10
        self.assertAlmostEqual(p1.forces.x, 10.0)


class TestGravitationalAttraction(unittest.TestCase):
    """Test Gravitational Attraction"""
    
    def test_gravitational_attraction(self):
        p1 = Particle(mass=1e10, position=Vector(0, 0, 0))
        p2 = Particle(mass=1e10, position=Vector(1, 0, 0))
        
        grav = GravitationalAttraction(G=6.674e-11)
        grav.apply(p1, [p1, p2])
        
        # F = G * m1 * m2 / r^2
        expected_force = 6.674e-11 * 1e10 * 1e10 / 1.0
        self.assertAlmostEqual(p1.forces.x, expected_force, delta=1e-5)


class TestPhysicsSimulation(unittest.TestCase):
    """Test Physics Simulation"""
    
    def test_add_particle(self):
        sim = PhysicsSimulation()
        p = Particle(mass=1.0)
        sim.add_particle(p)
        self.assertEqual(len(sim.particles), 1)
    
    def test_add_force(self):
        sim = PhysicsSimulation()
        gravity = Gravity()
        sim.add_force(gravity)
        self.assertEqual(len(sim.forces), 1)
    
    def test_step(self):
        sim = PhysicsSimulation(dt=0.01)
        p = Particle(mass=1.0, position=Vector(0, 0, 0))
        sim.add_particle(p)
        sim.add_force(Gravity(g=9.81))
        
        initial_time = sim.time
        sim.step()
        
        self.assertEqual(sim.time, initial_time + 0.01)
    
    def test_energy_conservation(self):
        # Test with spring system (approximate conservation)
        sim = PhysicsSimulation(dt=0.001)
        
        p1 = Particle(mass=1.0, position=Vector(0, 0, 0), velocity=Vector(1, 0, 0))
        p2 = Particle(mass=1.0, position=Vector(2, 0, 0), velocity=Vector(-1, 0, 0))
        
        sim.add_particle(p1)
        sim.add_particle(p2)
        
        initial_energy = sim.total_energy()
        
        # Run for a short time
        sim.run(100)
        
        final_energy = sim.total_energy()
        
        # Energy should be approximately conserved (some numerical error expected)
        self.assertAlmostEqual(initial_energy, final_energy, delta=0.5)


if __name__ == '__main__':
    unittest.main()
