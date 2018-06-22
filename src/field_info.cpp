#include "field_info.h"
#include "except.h"

field_info::field_info(int isize, int jsize, int ksize, int ilayout, int jlayout, int klayout, int halo, int alignment)
    : m_isize(isize), m_jsize(jsize), m_ksize(ksize), m_ilayout(ilayout), m_jlayout(jlayout), m_klayout(klayout),
      m_halo(halo), m_alignment(alignment),
      m_data_offset(((m_halo + m_alignment - 1) / m_alignment) * m_alignment - m_halo) {

    if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
        throw ERROR("invalid domain size");
    if (m_halo < 0)
        throw ERROR("invalid halo size");
    if (m_alignment <= 0)
        throw ERROR("invalid alignment");

    int ish = m_isize + 2 * m_halo;
    int jsh = m_jsize + 2 * m_halo;
    int ksh = m_ksize + 2 * m_halo;

    int s = 1;
    if (m_ilayout == 2) {
        m_istride = s;
        s *= ish;
    } else if (m_jlayout == 2) {
        m_jstride = s;
        s *= jsh;
    } else if (m_klayout == 2) {
        m_kstride = s;
        s *= ksh;
    } else {
        throw ERROR("invalid layout");
    }

    s = ((s + m_alignment - 1) / m_alignment) * m_alignment;

    if (m_ilayout == 1) {
        m_istride = s;
        s *= ish;
    } else if (m_jlayout == 1) {
        m_jstride = s;
        s *= jsh;
    } else if (m_klayout == 1) {
        m_kstride = s;
        s *= ksh;
    } else {
        throw ERROR("invalid layout");
    }

    if (m_ilayout == 0) {
        m_istride = s;
        s *= ish;
    } else if (m_jlayout == 0) {
        m_jstride = s;
        s *= jsh;
    } else if (m_klayout == 0) {
        m_kstride = s;
        s *= ksh;
    } else {
        throw ERROR("invalid layout");
    }

    m_storage_size = m_data_offset + s;
}
