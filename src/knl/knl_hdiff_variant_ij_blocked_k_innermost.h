#pragma once

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class hdiff_variant_ij_blocked_k_innermost final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            hdiff_variant_ij_blocked_k_innermost(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            void hdiff() override;

          private:
            int m_iblocksize, m_jblocksize;
        };

        extern template class hdiff_variant_ij_blocked_k_innermost<knl, float>;
        extern template class hdiff_variant_ij_blocked_k_innermost<knl, double>;

    } // namespace knl

} // namespace platform
