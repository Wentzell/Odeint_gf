#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <boost/multi_array.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>

using extent_range = boost::multi_array_types::extent_range;
using extent_gen = boost::multi_array_types::extent_gen;

inline extent_range ffreq( int n )
{
   return extent_range( -n, n );   // range [ -n , n )
}

inline extent_range bfreq( int n )
{
   return extent_range( -n, n + 1 );   // range [ -n , n + 1 )
}

   template <class T, std::size_t N>
std::ostream& operator<<( std::ostream& os, const std::array<T, N>& arr )
{
   std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(os, " "));
   return os;
}
/********************* Index type ********************/

template< unsigned int ndims_ >
class idx_obj_t							///< Index type that will be associated to each container
{
   public:
      static constexpr unsigned int ndims = ndims_; 				///< The number of dimensions

      using arr_t = std::array<int, ndims>; 	///< Array type

      arr_t idx_arr; 				///< Data array

      template< typename integer_t >
	 int& operator()( integer_t val )		///< Accesss by name, e.g. idx(name_t::w)
	 {
	    return idx_arr[int(val)]; 
	 }

      template< typename integer_t >
	 int operator()( integer_t val) const	///< Accesss by name, e.g. idx(name_t::w)
	 {
	    return idx_arr[int(val)]; 
	 }

      inline int* data()			///< Pointer to first data element
      {
	 return idx_arr.data(); 
      }

      idx_obj_t( const arr_t& idx_arr_ ):
	 idx_arr( idx_arr_ )	 
   {}

      idx_obj_t( arr_t&& idx_arr_ ):
	 idx_arr( idx_arr_ )	 
   {}

      idx_obj_t():
	 idx_arr()	 
   {}

      friend std::ostream& operator<<( std::ostream& os, idx_obj_t idx )	///< Output idx_t object to ostream
      {
	 return os << idx.idx_arr;
      }

}; 

/********************* Container ********************/

template< typename value_t_, unsigned int ndims_>
class gf: public boost::multi_array<value_t_, ndims_>
{
   public:

      // -------  Typedefs
      using value_t = value_t_; 						///< Type of container values
      using base_t = boost::multi_array<value_t, ndims_>; 			///< Type of base class
      using extents_t = boost::detail::multi_array::extent_gen<ndims_>; 	///< Type of extents object passed at contruction
      using idx_t = idx_obj_t<ndims_>; 
      using type = gf< value_t_, ndims_ >; 

      // -------- Member variables
      const typename base_t::size_type* shape_arr; 			///< Array containing the extent for each dimension
      const typename base_t::index* stride_arr; 			///< Array containing the stride for each dimension
      const typename base_t::index* idx_bases; 				///< Array containing the base index for each dimension
      static constexpr unsigned int ndims = ndims_; 			///< The number of dimensions


      inline int get_pos_1d( const idx_t& idx ) const			///< For a given idx_t object, returns the corresponding position in a 1d array
      {
	 return get_pos_1d( idx.idx_arr ); 
      }

      int get_pos_1d( const std::array<int, ndims>& idx_arr ) const	///< For a given index array, returns the corresponding position in a 1d array
      {
	 int val = 0; 
	 for( int i = 0; i < ndims; ++i )
	 {
	    val += stride_arr[i]*( idx_arr[i] - idx_bases[i] ); 
	 }
	 return val; 
      }

      idx_t get_idx( int pos_1d ) const					///< For a given index in a 1d array, returns the corresponding idx_t object
      {
	 assert( pos_1d < base_t::num_elements() ); 
	 std::array<int, ndims> idx_arr; 

	 for( int i = 0; i < ndims; ++i )
	 {
	    idx_arr[i] = idx_bases[i] + pos_1d / stride_arr[i];
	    pos_1d -= ( idx_arr[i] - idx_bases[i] ) * stride_arr[i]; 
	 }

	 return idx_t( idx_arr ); 
      }

      void fill_idx_lst( std::vector<idx_t>& idx_lst )	const		///< Fills a std::vector<idx_t> with all possible sets of indeces
      {
	 for( int i = 0; i < base_t::num_elements(); ++i )
	    idx_lst.push_back( get_idx( i ) ); 
      }

      typedef value_t (*init_func_t)( const idx_t& idx ); 		///< Initalization function type that returns value_t for a given idx_t object

      void init( init_func_t init_func )				///< Initializes values with a given initialization function
      {
	 for( int i = 0; i < base_t::num_elements(); ++i )
	 {
	    idx_t idx( get_idx( i ) ) ; 
	    operator()( idx ) = init_func( idx ); 
	 }
      }

      value_t& operator()( const idx_t& idx )				///< Return value of container for given index object
      {
	 return data_ptr[ get_pos_1d( idx.idx_arr ) ]; 
      }

      const value_t& operator()( const idx_t& idx ) const		///< Return value of container for given index object
      {
	 return data_ptr[ get_pos_1d( idx.idx_arr ) ]; 
      }

      value_t& operator()( int i )					///< Return value of container for given index object
      {
	 return data_ptr[ i ]; 
      }

      const value_t& operator()( int i ) const				///< Return value of container for given index object
      {
	 return data_ptr[ i ]; 
      }

      inline value_t* data()						///< Pointer to first data element
      {
	 return base_t::data(); 
      }

      gf( extents_t idx_ranges ):
	 base_t( idx_ranges ), shape_arr( base_t::shape() ), stride_arr( base_t::strides() ), idx_bases( base_t::index_bases() ), data_ptr( data() ) 
   {};

      gf( const base_t& boost_arr ):
	 base_t( boost_arr ), shape_arr( boost_arr.shape() ), stride_arr( boost_arr.strides() ), idx_bases( boost_arr.index_bases() ), data_ptr( data() ) 
   {};

      gf( const type& gf_obj ):
	 base_t( static_cast< const base_t& >(gf_obj) ), shape_arr( gf_obj.shape_arr ), stride_arr( gf_obj.stride_arr ), idx_bases( gf_obj.idx_bases ), data_ptr( data() ) 
   {};

      //type& operator =( const type& gf_obj )
      //{
	 //base_t::operator=( static_cast< const base_t& >(gf_obj) );
	 //shape_arr = gf_obj.shape_arr;
	 //stride_arr = gf_obj.stride_arr;
	 //idx_bases = gf_obj.idx_bases; 
      //};
	 
      //gf( base_t&& boost_arr ):
	 //base_t( boost_arr ), shape_arr( boost_arr.shape() ), stride_arr( boost_arr.strides() ), idx_bases( boost_arr.index_bases() ), data_ptr( data() ) 
   //{};

      //gf( type&& gf_obj ):
	 //base_t( static_cast<base_t&&> gf_obj ), shape_arr( gf_obj.shape_arr ), stride_arr( gf_obj.stride_arr ), idx_bases( gf_obj.idx_bases ), data_ptr( data() ) 
   //{};

   private:
      value_t* data_ptr; 
};

template< typename gf_t_ >
struct is_gf : 
   boost::mpl::false_ { };  

template< typename value_t_, unsigned int ndims_ >
struct is_gf< gf< value_t_, ndims_ > > :
boost::mpl::true_ { };  

   //template< typename gf_t_ >
//typename boost::enable_if< is_gf< gf_t_ >, gf_t_ >::type abs( const gf_t_& lhs )

template< typename scalar_t_ >
struct is_scalar : 
   boost::mpl::false_ { };  

template<> struct is_scalar<double> : boost::mpl::true_ { };  
template<> struct is_scalar<int> : boost::mpl::true_ { };  
template<> struct is_scalar<std::complex<double>> : boost::mpl::true_ { };  


// --------------- Abs and Norm

   template< typename value_t_, unsigned int ndims_ >
gf< value_t_, ndims_ > abs( const gf< value_t_, ndims_ >& lhs )
{
   using std::abs; 

   gf< value_t_, ndims_ > res( lhs ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      res(i) = abs(res(i)); 
   return res; 
}

   template< typename value_t_, unsigned int ndims_ >
double norm( const gf<value_t_,ndims_>& lhs )
{
   gf<value_t_,ndims_> gf_abs = abs( lhs ); 
   return *( std::max_element( gf_abs.origin(), gf_abs.origin() + gf_abs.num_elements() ) ); 
}


// -------------- OPERATORS 

// ------ Single gf operations

/// -gf
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_> operator-( const gf< value_t_, ndims_ >& lhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      res(i) = -res(i); 
   return res; 
}

// ------ Two gf operations

/// gf1 += gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_>& operator+=( gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   assert( lhs.num_elements() == rhs.num_elements() && " Adding gf's of different size " ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) += rhs(i); 
   return lhs; 
}

/// gf1 + gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_> operator+( const gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res += rhs; 
   return res; 
}

/// gf1 -= gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_>& operator-=( gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   assert( lhs.num_elements() == rhs.num_elements() && " Adding gf's of different size " ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) -= rhs(i); 
   return lhs; 
}

/// gf1 - gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_> operator-( const gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res -= rhs; 
   return res; 
}

/// gf1 *= gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_>& operator*=( gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   assert( lhs.num_elements() == rhs.num_elements() && " Adding gf's of different size " ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) *= rhs(i); 
   return lhs; 
}

/// gf1 * gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_> operator*( const gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res *= rhs;  
   return res; 
}

/// gf1 /= gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_>& operator/=( gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   assert( lhs.num_elements() == rhs.num_elements() && " Adding gf's of different size " ); 
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) /= rhs(i); 
   return lhs; 
}

/// gf1 / gf2
   template< typename value_t_, unsigned int ndims_ >
gf<value_t_,ndims_> operator/( const gf< value_t_, ndims_ >& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res /= rhs; 
   return res; 
}


// ------------ Scalar operations

/// gf += s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_>& >::type 
operator+=( gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) += rhs; 
   return lhs; 
}

/// gf + s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator+( const gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res += rhs; 
   return res; 
}

/// s + gf
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator+( const scalar_t_& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( rhs ); 
   res += lhs; 
   return res; 
}

/// gf -= s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_>& >::type 
operator-=( gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) -= rhs; 
   return lhs; 
}

/// gf - s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator-( const gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res -= rhs; 
   return res; 
}

/// s - gf
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator-( const scalar_t_& lhs,  const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( -rhs ); 
   res += rhs; 
   return res; 
}

/// gf *= s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_>& >::type 
operator*=( gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) *= rhs; 
   return lhs; 
}

/// gf * s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator*( const gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res *= rhs; 
   return res; 
}

/// s * gf
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator*( const scalar_t_& lhs, const gf< value_t_, ndims_ >& rhs )
{
   gf< value_t_, ndims_ > res( rhs ); 
   res *= lhs; 
   return res; 
}

/// gf /= s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_>& >::type 
operator/=( gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   for( int i = 0; i < lhs.num_elements(); i++ )
      lhs(i) /= rhs; 
   return lhs; 
}

/// gf / s
   template< typename value_t_, unsigned int ndims_, typename scalar_t_ >
typename boost::enable_if< is_scalar< scalar_t_ >, gf<value_t_,ndims_> >::type 
operator/( const gf< value_t_, ndims_ >& lhs, const scalar_t_& rhs )
{
   gf< value_t_, ndims_ > res( lhs ); 
   res /= rhs; 
   return res; 
}
