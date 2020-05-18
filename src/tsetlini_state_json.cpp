#include "tsetlini_state_json.hpp"
#include "tsetlini_state.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_params.hpp"
#include "tsetlini_classifier_state_private.hpp"
#include "params_companion.hpp"

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


void to_json(json & j, Tsetlini::ClassifierState::ta_state_v_type const & p)
{
    json ta_state;

    ta_state["width"] = std::visit([](auto const & p)
        {
            using value_type = typename std::decay<decltype(p)>::type::value_type;
            return sizeof (value_type);
        }, p);
    ta_state["data"] = json::array_t{};

    std::visit([&](auto const & p)
        {
            auto const [nr, nc] = p.shape();

            for (auto rit = 0u; rit < nr; ++rit)
            {
                // ugly, but as for now json cannot be easily created
                // from a range of pointers
                auto jrow = json(nc, 0);

                auto row_data = p.row_data(rit);
                std::copy(row_data, row_data + nc, jrow.begin());

                ta_state["data"].push_back(jrow);
            }
        }, p);

    j = ta_state;
}


static void from_json(json const & j, Tsetlini::ClassifierState::ta_state_v_type & p)
{
    std::size_t width = 0u;
    j.at("width").get_to(width);

    auto const nr = j["data"].size();
    auto const nc = j["data"][0].size();

    if (width == 1)
    {
        p = Tsetlini::numeric_matrix_int8(nr, nc);
    }
    else if (width == 2)
    {
        p = Tsetlini::numeric_matrix_int16(nr, nc);
    }
    else // if (width == 4)
    {
        p = Tsetlini::numeric_matrix_int32(nr, nc);
    }

    std::visit([&](auto & p)
        {
            auto const & data = j["data"];

            for (auto rit = 0u; rit < nr; ++rit)
            {
                auto row_data = p.row_data(rit);

                std::copy(data[rit].cbegin(), data[rit].cend(), row_data);
            }
        }, p);
}


namespace nlohmann
{


template<>
struct adl_serializer<Tsetlini::param_value_t>
{
    static void to_json(json & j, Tsetlini::param_value_t const & p)
    {
        if (std::holds_alternative<int>(p))
        {
            j = std::get<int>(p);
        }
        else if (std::holds_alternative<Tsetlini::seed_type>(p))
        {
            j = std::get<Tsetlini::seed_type>(p);
        }
        else if (std::holds_alternative<Tsetlini::real_type>(p))
        {
            j = std::get<Tsetlini::real_type>(p);
        }
        else if (std::holds_alternative<bool>(p))
        {
            j = std::get<bool>(p);
        }
        else if (std::holds_alternative<std::string>(p))
        {
            j = std::get<std::string>(p);
        }
        else if (std::holds_alternative<Tsetlini::none_type>(p))
        {
            j = nullptr;
        }
        else
        {
            // TODO?
            j = nullptr;
        }
    }

    static void from_json(json const & j, Tsetlini::param_value_t & p)
    {
        if (j.is_boolean())
        {
            p = j.get<bool>();
        }
        else if (j.is_number_unsigned())
        {
            p = j.get<Tsetlini::seed_type>();
        }
        else if (j.is_number_float())
        {
            p = j.get<Tsetlini::real_type>();
        }
        else if (j.is_number_integer())
        {
            p = j.get<int>();
        }
        else if (j.is_string())
        {
            p = j.get<std::string>();
        }
        else if (j.is_null())
        {
            p = std::nullopt;
        }
    }
};

} // namespace nlohmann


namespace Tsetlini
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
    state.ta_state = js.at("ta_state").get<Tsetlini::ClassifierState::ta_state_v_type>();
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
    // end-of-hack

    reset_state_cache(state);
}


} // namespace Tsetlini
