#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, int, typename, typename, typename...>
	class MapReduceImpl;
	
	template<int GivenArity, typename RetType, typename RedType, typename... Args>
	MapReduceImpl<resolve_map_arity<GivenArity, Args...>::value, GivenArity, RetType, RedType, Args...>
	MapReduceWrapper(std::function<RetType(Args...)> map, std::function<RedType(RedType, RedType)> red)
	{
		return MapReduceImpl<resolve_map_arity<GivenArity, Args...>::value, GivenArity, RetType, RedType, Args...>(map, red);
	}
	
	// For function pointers
	template<int GivenArity = SKEPU_UNSET_ARITY, typename RetType, typename RedType, typename... Args>
	auto MapReduce(RetType(*map)(Args...), RedType(*red)(RedType, RedType))
		-> decltype(MapReduceWrapper<GivenArity>((std::function<RetType(Args...)>)map, (std::function<RedType(RedType, RedType)>)red))
	{
		return MapReduceWrapper<GivenArity>((std::function<RetType(Args...)>)map, (std::function<RedType(RedType, RedType)>)red);
	}
	
	// For lambdas and functors
	template<int GivenArity = SKEPU_UNSET_ARITY, typename T1, typename T2>
	auto MapReduce(T1 map, T2 red)
		-> decltype(MapReduceWrapper<GivenArity>(lambda_cast(map), lambda_cast(red)))
	{
		return MapReduceWrapper<GivenArity>(lambda_cast(map), lambda_cast(red));
	}
	
	
	/* MapReduce "semantic guide" for the SkePU 2 precompiler.
	 * Sequential implementation when used with any C++ compiler.
	 * Works with any number of variable arguments > 0.
	 * Works with any number of constant arguments >= 0.
	 */
	template<int InArity, int GivenArity, typename RetType, typename RedType, typename... Args>
	class MapReduceImpl: public SeqSkeletonBase
	{
		using MapFunc = std::function<RetType(Args...)>;
		using RedFunc = std::function<RedType(RedType, RedType)>;
		
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t OutArity = out_size<RetType>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0);
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		using defaultDim = typename std::conditional<indexed, index_dimension<typename first_element<Args...>::type>, std::integral_constant<int, 1>>::type;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<InArity, 0>::type elwise_indices{};
		static constexpr typename make_pack_indices<InArity + anyCont, InArity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, InArity + anyCont>::type const_indices{};
		
		using F = ConditionalIndexForwarder<indexed, MapFunc>;
		
		
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		RetType apply(pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, size_t size, CallArgs&&... args)
		{
			if (disjunction((get<EI>(args...).size() < size)...))
				SKEPU_ERROR("MapReduce: Non-matching container sizes");
			
			RetType res = this->m_start;
			auto elwiseIterators = std::make_tuple(get<EI>(args...).begin()...);
			
			while (size --> 0)
			{
				auto index = std::get<0>(elwiseIterators).getIndex();
				RetType temp = F::forward(mapFunc, index,
					*std::get<EI>(elwiseIterators)++...,
					get<AI>(args...).hostProxy()...,
					get<CI>(args...)...
				);
				pack_expand((get_or_return<OI>(res) = redFunc(get_or_return<OI>(res), get_or_return<OI>(temp)), 0)...);
			}
			
			return res;
		}
		
		template<size_t... OI, size_t... AI, size_t... CI, typename... CallArgs>
		RetType zero_apply(pack_indices<OI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size = this->default_size_i;
			if (defaultDim::value >= 2) size *= this->default_size_j;
			if (defaultDim::value >= 3) size *= this->default_size_k;
			if (defaultDim::value >= 4) size *= this->default_size_l;
			
			RetType res = this->m_start;
			for (size_t i = 0; i < size; ++i)
			{
				RetType temp = F::forward(mapFunc,
					make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l),
					get<AI>(args...).hostProxy()...,
					get<CI>(args...)...
				);
				pack_expand((get<OI>(res) = redFunc(get<OI>(res), get_or_return<OI>(temp)), 0)...);
			}
			return res;
		}
		
		
	public:
		
		void setStartValue(RetType val)
		{
			this->m_start = val;
		}
		
		void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
		{
			this->default_size_i = i;
			this->default_size_j = j;
			this->default_size_k = k;
			this->default_size_l = l;
		}
		
		// For first elwise argument as container
		template<typename First, template<class> class Container, typename... CallArgs, REQUIRES_VALUE(is_skepu_container<Container<First>>)>
		RetType operator()(Container<First>& arg1, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) + 1 == numArgs, "Number of arguments not matching Map function");
			return apply(out_indices, elwise_indices, any_indices, const_indices, arg1.size(), arg1.begin(), std::forward<CallArgs>(args)...);
		}
		
		// For first elwise argument as iterator
		template<typename Iterator, typename... CallArgs, REQUIRES(sizeof...(CallArgs) + 1 == numArgs)>
		RetType operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args)
		{
			return apply(out_indices, elwise_indices, any_indices, const_indices, arg1_end - arg1, arg1, std::forward<CallArgs>(args)...);
		}
		
		// For no elwise arguments
		template<typename... CallArgs, REQUIRES(sizeof...(CallArgs) == numArgs)>
		RetType operator()(CallArgs&&... args)
		{
			return this->zero_apply(out_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
		}
		
	private:
		MapFunc mapFunc;
		RedFunc redFunc;
		MapReduceImpl(MapFunc map, RedFunc red): mapFunc(map), redFunc(red) {}
		
		RetType m_start{};
		size_t default_size_i = 0;
		size_t default_size_j = 0;
		size_t default_size_k = 0;
		size_t default_size_l = 0;
		
		friend MapReduceImpl<InArity, GivenArity, RetType, RedType, Args...> MapReduceWrapper<GivenArity, RetType, RedType, Args...>(MapFunc, RedFunc);
		
	}; // end class MapReduce
	
} // end namespace skepu
