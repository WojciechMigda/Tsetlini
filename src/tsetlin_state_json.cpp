#include "tsetlin_state_json.hpp"
#include "tsetlin_state.hpp"
#include "tsetlin_types.hpp"
#include "tsetlin_params.hpp"

#include "json.hpp"

#include <string>
#include <iterator>
#include <vector>
#include <optional>


using json = nlohmann::json;


static void to_json(json & j, IRNG const & p)
{
    std::vector<IRNG::value_type> res(p.RESp(), p.RESp() + p.MTSZ * p.NS);
    std::vector<unsigned int> mt(p.MTp(), p.MTp() + p.MTSZ * p.NS);

    j = json{{"index", p.index}, {"RES", res}, {"MT", mt}};
}


static void from_json(json const & j, IRNG & p)
{
    j.at("index").get_to(p.index);

    std::vector<IRNG::value_type> res;
    j.at("RES").get_to(res);
    std::copy(res.cbegin(), res.cend(), p.RESp());

    std::vector<unsigned int> mt;
    j.at("MT").get_to(mt);
    std::copy(mt.cbegin(), mt.cend(), p.MTp());
}


static void to_json(json & j, FRNG const & p)
{
    std::vector<FRNG::value_type> res(p.RESp(), p.RESp() + p.MTSZ * p.NS);
    std::vector<unsigned int> mt(p.MTp(), p.MTp() + p.MTSZ * p.NS);

    j = json{{"index", p.index}, {"RES", res}, {"MT", mt}};
}


static void from_json(json const & j, FRNG & p)
{
    j.at("index").get_to(p.index);

    std::vector<FRNG::value_type> res;
    j.at("RES").get_to(res);
    std::copy(res.cbegin(), res.cend(), p.RESp());

    std::vector<unsigned int> mt;
    j.at("MT").get_to(mt);
    std::copy(mt.cbegin(), mt.cend(), p.MTp());
}


static void to_json(json & j, std::vector<Tsetlin::aligned_vector_int> const & p)
{
    json ta_state;
    std::transform(p.cbegin(), p.cend(), std::back_inserter(ta_state), [](auto const & v){ return json(v); });

    j = ta_state;
}


static void from_json(json const & j, std::vector<Tsetlin::aligned_vector_int> & p)
{
    p.resize(j.size());
    std::transform(j.cbegin(), j.cend(), p.begin(), [](json const & j){ return Tsetlin::aligned_vector_int(j.cbegin(), j.cend()); });
}


namespace nlohmann
{


template<>
struct adl_serializer<Tsetlin::param_value_t>
{
    static void to_json(json & j, Tsetlin::param_value_t const & p)
    {
        if (std::holds_alternative<int>(p))
        {
            j = std::get<int>(p);
        }
        else if (std::holds_alternative<Tsetlin::seed_type>(p))
        {
            j = std::get<Tsetlin::seed_type>(p);
        }
        else if (std::holds_alternative<Tsetlin::real_type>(p))
        {
            j = std::get<Tsetlin::real_type>(p);
        }
        else if (std::holds_alternative<bool>(p))
        {
            j = std::get<bool>(p);
        }
        else if (std::holds_alternative<Tsetlin::none_type>(p))
        {
            j = nullptr;
        }
        else
        {
            // TODO?
            j = nullptr;
        }
    }

    static void from_json(json const & j, Tsetlin::param_value_t & p)
    {
        if (j.is_boolean())
        {
            p = j.get<bool>();
        }
        else if (j.is_number_unsigned())
        {
            p = j.get<Tsetlin::seed_type>();
        }
        else if (j.is_number_float())
        {
            p = j.get<Tsetlin::real_type>();
        }
        else if (j.is_number_integer())
        {
            p = j.get<int>();
        }
        else if (j.is_null())
        {
            p = std::nullopt;
        }
    }
};

} // namespace nlohmann


namespace Tsetlin
{


std::string to_json_string(ClassifierState const & state)
{
    json js;

    js["ta_state"] = state.ta_state;
    js["igen"] = state.igen;
    js["fgen"] = state.fgen;
    js["params"] = state.m_params;

    return js.dump();
}


void from_json_string(ClassifierState & state, std::string const & jss)
{
    auto js = json::parse(jss);

    state.igen = js.at("igen").get<IRNG>();
    state.fgen = js.at("fgen").get<FRNG>();
    state.ta_state = js.at("ta_state").get<std::vector<Tsetlin::aligned_vector_int>>();
    state.m_params = js.at("params").get<params_t>();

    // So, we need a hack, since stringified json doesn't distinguish
    // between signed and unsigned types for integer values > 0.
    // most of our params are signed integers, as wrapped
    // by std::variant, except for "random_state".
    // Json parser will report them as unsigned.
    // Here we will re-cast those integers as signed (contrary to json
    // enumeration).
    for (auto & [k, v]: state.m_params)
    {
        if (std::holds_alternative<seed_type>(v) and k != "random_state")
        {
            state.m_params[k] = static_cast<int>(std::get<seed_type>(v));
        }
    }
}


} // namespace Tsetlin
