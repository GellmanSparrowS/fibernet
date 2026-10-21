"""Independent constitutive contracts and analytical convergence benchmarks."""
import json
import math
import unittest
import numpy as np
from fibernet.core.viscoelastic import LinearViscoelasticMaterial as Law


class MaterialContracts(unittest.TestCase):
    def test_reject_invalid_passive_parameters(self):
        for args in ((0.,()),(-1.,()),(1.,((0.,1.),)),(1.,((1.,0.),)),
                     (float('nan'),()),(True,()),(1.,((1.,float('inf')),))):
            with self.subTest(args=args),self.assertRaises(ValueError): Law(*args)

    def test_named_families_and_json(self):
        laws=(Law.maxwell(3.,6.),Law.kelvin_voigt(3.,6.),
              Law.standard_linear_solid(1.,2.,3.),Law(1.,((2.,3.),(4.,5.))))
        for law in laws:
            self.assertEqual(law,Law.from_dict(json.loads(json.dumps(law.to_dict()))))
        self.assertEqual(laws[0].branches,((3.,2.),))
        self.assertEqual(laws[1].parallel_viscosity,6.)

    def test_trial_no_mutation_and_tangent(self):
        law=Law(2.,((3.,.5),(4.,2.)),.1)
        old=np.array([.01,-.02]); v=np.array([[.002,-.01],[.004,.003]])
        old_copy=old.copy(); v_copy=v.copy(); new=np.array([.03,.01]); dt=.1
        out=law.trial(new,old,v,dt)
        np.testing.assert_array_equal(old,old_copy); np.testing.assert_array_equal(v,v_copy)
        expected=2+3/(1+.1/.5)+4/(1+.1/2)+.1/.1
        self.assertAlmostEqual(out['tangent'],expected)
        perturbed=law.trial(new+1e-7,old,v,dt)
        np.testing.assert_allclose((perturbed['stress']-out['stress'])/1e-7,expected,rtol=1e-9)
        out['viscous_strain'][0,0]=99
        np.testing.assert_array_equal(v,v_copy)

    def test_physical_numerical_energy_separate(self):
        for law in (Law(2.),Law.maxwell(3.,.6),Law.kelvin_voigt(3.,.6),Law(2.,((3.,.5),),.2)):
            old=np.array([.1,-.2]); new=np.array([.12,-.1]); v=np.zeros((len(law.branches),2))
            out=law.trial(new,old,v,.02)
            np.testing.assert_allclose(out['endpoint_work_density'],
                out['stored_energy_density']-out['previous_stored_energy_density']+
                out['physical_dissipation_density']+out['numerical_dissipation_density'],rtol=1e-12,atol=1e-15)
            self.assertTrue(np.all(out['physical_dissipation_density']>=0))
            self.assertTrue(np.all(out['numerical_dissipation_density']>=0))

    def test_invalid_step_and_history(self):
        law=Law.maxwell(1.,1.)
        for dt in (0.,-1.,float('inf'),True,1e8):
            with self.subTest(dt=dt),self.assertRaises((ValueError,FloatingPointError)):
                law.trial([0.],[0.],[[0.]],dt)
        with self.assertRaises(ValueError): law.trial([0.],[0.],[],.1)

    def test_finite_ramp_relaxation_refinement(self):
        # Expected values derived directly from exponential convolution, not trial().
        law=Law(1e6,((2e6,.1),(1e6,.5)))
        errors=[]; amplitude=.002; ramp=.2
        for n in (50,100,200):
            times=np.linspace(0.,1.,n+1); strain=amplitude*np.minimum(times/ramp,1.)
            expected=law.equilibrium_modulus*strain
            for e,tau in law.branches:
                during=e*amplitude/ramp*tau*(-np.expm1(-times/tau))
                after=e*amplitude/ramp*tau*(-math.expm1(-ramp/tau))*np.exp(-(times-ramp)/tau)
                expected=expected+np.where(times<=ramp,during,after)
            v=np.zeros((2,)); old=0.; values=[0.]
            for k in range(1,len(times)):
                out=law.trial(strain[k],old,v,times[k]-times[k-1])
                old=strain[k];v=out['viscous_strain'];values.append(float(out['stress']))
            errors.append(float(np.max(np.abs(np.array(values)-expected)))/(law.instantaneous_spring_modulus*amplitude))
        self.assertLess(errors[-1],.01)
        self.assertLess(errors[1],.65*errors[0]); self.assertLess(errors[2],.65*errors[1])

    def test_cyclic_loss_against_complex_modulus(self):
        law=Law(1e6,((2e6,.05),(1e6,.2)),1000.)
        omega=2*math.pi; amplitude=.001
        storage=law.equilibrium_modulus+sum(e*(omega*tau)**2/(1+(omega*tau)**2) for e,tau in law.branches)
        loss=omega*law.parallel_viscosity+sum(e*omega*tau/(1+(omega*tau)**2) for e,tau in law.branches)
        errors=[]
        for per_cycle in (40,80,160):
            t=np.arange(10*per_cycle+1)/per_cycle; strain=amplitude*np.sin(omega*t)
            v=np.zeros((2,)); old=0.; stress=[0.]; physical=[]
            for k in range(1,len(t)):
                out=law.trial(strain[k],old,v,t[k]-t[k-1]); old=strain[k];v=out['viscous_strain']
                stress.append(float(out['stress']));physical.append(float(out['physical_dissipation_density']))
            tt=t[-per_cycle:]; yy=np.array(stress[-per_cycle:])/amplitude
            recovered=np.linalg.lstsq(np.column_stack((np.sin(omega*tt),np.cos(omega*tt))),yy,rcond=None)[0]
            cycle_loss=sum(physical[-per_cycle:]); expected_loss=math.pi*amplitude**2*loss
            errors.append(max(abs(recovered[0]/storage-1),abs(recovered[1]/loss-1),abs(cycle_loss/expected_loss-1)))
        self.assertLess(errors[-1],.05)
        self.assertLess(errors[1],.8*errors[0]);self.assertLess(errors[2],.8*errors[1])


if __name__=='__main__': unittest.main()
