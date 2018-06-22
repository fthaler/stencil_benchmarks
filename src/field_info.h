#pragma once

class field_info {
  public:
    field_info(int isize, int jsize, int ksize, int ilayout, int jlayout, int klayout, int halo, int alignment);

    int isize() const { return m_isize; }
    int jsize() const { return m_jsize; }
    int ksize() const { return m_ksize; }

    int ilayout() const { return m_ilayout; }
    int jlayout() const { return m_jlayout; }
    int klayout() const { return m_klayout; }

    int istride() const { return m_istride; }
    int jstride() const { return m_jstride; }
    int kstride() const { return m_kstride; }

    int halo() const { return m_halo; }
    int alignment() const { return m_alignment; }

    int storage_size() const { return m_storage_size; }
    int data_offset() const { return m_data_offset; }

    int index(int i, int j, int k) const { return i * m_istride + j * m_jstride + k * m_kstride; }
    int last_index() const { return index(m_isize - 1, m_jsize - 1, m_ksize - 1); }
    int zero_offset() const { return m_data_offset + index(m_halo, m_halo, m_halo); }

  private:
    int m_isize, m_jsize, m_ksize;
    int m_ilayout, m_jlayout, m_klayout;
    int m_istride, m_jstride, m_kstride;
    int m_halo, m_alignment;
    int m_data_offset, m_storage_size;
};
