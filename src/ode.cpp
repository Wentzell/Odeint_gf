#include <iostream>
#include <complex>

#include <boost/numeric/odeint.hpp>

#include <arithmetic_tuple.h>
#include <gf.h>

using namespace ReaK; 
using dcomplex = std::complex< double >; 

const int N=100;//number of Matsubara frequencies

#define INSERT_COPY_AND_ASSIGN(X) 					\
X( const X & obj ):    							\
   base_t( obj )							\
{}       								\
X( X && obj ):								\
   base_t( std::move(obj) )						\
{}      								\
X & operator=( const X & obj )						\
{									\
   base_t::operator=( obj ); 						\
   return *this; 							\
} 									\
X & operator=( X && obj )						\
{									\
   base_t::operator=( std::move( obj ) ); 				\
   return *this; 							\
} 

enum class I1P{ w }; 
class gf_1p_t : public gf< dcomplex, 1 > 		///< Container type for one-particle correlation function
{
   public:
      using base_t = gf< dcomplex, 1 >; 

      gf_1p_t():
	 gf< dcomplex, 1 >( boost::extents[ffreq(N)] )
   {}
      INSERT_COPY_AND_ASSIGN(gf_1p_t)
}; 
using idx_1p_t = gf_1p_t::idx_t; 

enum class I2P{ W, w }; 
class gf_2p_t : public gf< dcomplex, 2 > 		///< Container type for two-particle correlation functions
{
   public:
      using base_t = gf< dcomplex, 2 >; 

      gf_2p_t():
	 gf< dcomplex, 2 >( boost::extents[bfreq(N)][ffreq(N)] )
   {}
      INSERT_COPY_AND_ASSIGN(gf_2p_t)
}; 
using idx_2p_t = gf_2p_t::idx_t; 

// The state type for the Ode solver, tuple of gf's with arithmetic operations
class state_t: public arithmetic_tuple< gf_1p_t, gf_2p_t > 
{
   public:
      using base_t = arithmetic_tuple< gf_1p_t, gf_2p_t >; 
      using Sig_t = gf_1p_t; 
      using Gam_t = gf_2p_t; 

      inline Sig_t& Sig() { return( std::get<0>( *this ) ); } 
      inline const Sig_t& Sig() const { return( std::get<0>( *this ) ); }

      inline Gam_t& Gam() { return( std::get<1>( *this ) ); } 
      inline const Gam_t& Gam() const { return( std::get<1>( *this ) ); }

      state_t():
	 base_t()
   {}
      INSERT_COPY_AND_ASSIGN(state_t)
}; 

// Norm of state_t, needed for adaptive stepping routines
namespace boost { namespace numeric { namespace odeint {
   template<>
      struct vector_space_norm_inf< state_t >
      {
	 typedef double result_type;
	 double operator()( const state_t &p ) const
	 {
	    using namespace std; 
	    return norm( p ); 
	 }
      };
}}}

// The rhs of x' = f(x) defined as a class 
class rhs_t{
   public:
      void operator()( const state_t &x , state_t &dxdt , const double  t  )
      {
	 std::cout << " Evaluation at scale " << t << std::endl; 
	 std::cout << " Tracking Gam0:  " << x.Gam()(0) << std::endl; 

	 dxdt.Sig().init( []( const idx_1p_t& idx )->double{ return 1.0; } );
	 dxdt.Gam().init( []( const idx_2p_t& idx )->double{ return 1.0; } );
      }
};

auto my_test( int a ) -> double { return a; }

int main(int /* argc */ , char** /* argv */ )
{
   using namespace boost::numeric::odeint;
   using namespace std; 

   state_t state_vec; 

   double a = 10.0; 

   // Initialize current state
   state_vec.Sig().init( []( const idx_1p_t& idx )->double{ return 1.1; } );
   state_vec.Gam().init( []( const idx_2p_t& idx )->double{ return 1.2; } );

   cout << " norm( state_vec ) " << norm( state_vec ) << endl; 

   cout << " Gam0 init " << state_vec.Gam()(0) << endl; 

   // Save copies of initial gfs

   // Some tests

   // instantiate rhs object
   rhs_t rhs;

   // Type of adaptive stepper, use vector_space_algebra here!
   typedef runge_kutta_cash_karp54< state_t, double, state_t, double, vector_space_algebra > error_stepper_t; 

   // Constants
   double ERR_ABS = 0.01; 
   double ERR_REL = 0.01; 

   double LAM_START = 0.0; 
   double LAM_FIN = 1.0; 
   double INIT_STEP = 0.1; 

   // Integrate ODE 
   int steps = integrate_adaptive( make_controlled< error_stepper_t >( ERR_ABS, ERR_REL ), rhs, state_vec, LAM_START, LAM_FIN, INIT_STEP ); 
   //error_stepper_t stepper; 
   //int steps = integrate_const( stepper, rhs, state_vec, LAM_START, LAM_FIN, INIT_STEP ); 

   // Output results
   cout << " Gam0 final " << state_vec.Gam()(0) << endl; 
}
