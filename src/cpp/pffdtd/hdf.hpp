// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "pffdtd/assert.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/mdspan.hpp"
#include "pffdtd/utility.hpp"

#include "hdf5.h"
#include <fmt/format.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace pffdtd {

template<typename T>
inline constexpr auto isStdVector = false;

template<typename T>
inline constexpr auto isStdVector<std::vector<T>> = true;

struct HDF5Reader {
  explicit HDF5Reader(std::filesystem::path const& path)
      : _handle{[&] {
        if (not std::filesystem::exists(path)) {
          raisef<std::invalid_argument>("file '{}' does not exist", path.string());
        }
        return H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      }()} { }

  ~HDF5Reader() { H5Fclose(_handle); }

  HDF5Reader(HDF5Reader const& other) = delete;
  HDF5Reader(HDF5Reader&& other)      = delete;

  auto operator=(HDF5Reader const& other) -> HDF5Reader& = delete;
  auto operator=(HDF5Reader&& other) -> HDF5Reader&      = delete;

  [[nodiscard]] auto handle() const noexcept -> hid_t { return _handle; }

  template<typename T>
  [[nodiscard]] auto read(char const* dataset) -> T {
    if constexpr (isStdVector<T>) {
      return readBuffer<typename T::value_type>(dataset);
    } else if constexpr (std::is_same_v<T, bool>) {
      return static_cast<bool>(read<int8_t>(dataset));
    } else {
      auto set = H5Dopen(_handle, dataset, H5P_DEFAULT);
      if constexpr (std::is_same_v<T, int64_t>) {
        auto val  = T{};
        auto* ptr = static_cast<void*>(&val);
        auto err  = H5Dread(set, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
        checkErrorAndCloseDataset(dataset, set, err);
        return val;
      } else if constexpr (std::is_same_v<T, int8_t>) {
        auto val  = T{};
        auto* ptr = static_cast<void*>(&val);
        auto err  = H5Dread(set, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
        checkErrorAndCloseDataset(dataset, set, err);
        return val;
      } else if constexpr (std::is_same_v<T, double>) {
        auto val  = T{};
        auto* ptr = static_cast<void*>(&val);
        auto err  = H5Dread(set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
        checkErrorAndCloseDataset(dataset, set, err);
        return val;
      } else {
        static_assert(always_false<T>);
      }
    }

    return {};
  }

  template<typename T>
  [[nodiscard]] auto readBuffer(char const* dataset) -> std::vector<T> {
    auto set   = H5Dopen(_handle, dataset, H5P_DEFAULT);
    auto space = H5Dget_space(set);

    // auto ndims = 1;
    // PFFDTD_ASSERT(H5Sget_simple_extent_ndims(space) == ndims);

    auto ndims = H5Sget_simple_extent_ndims(space);
    auto dims  = std::array<hsize_t, 3>{};
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);

    auto size = ndims == 1 ? dims[0] : dims[0] * dims[1];

    if constexpr (std::is_same_v<T, int64_t>) {
      auto type = H5T_NATIVE_INT64;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      auto type = H5T_NATIVE_UINT8;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      auto type = H5T_NATIVE_INT8;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    } else if constexpr (std::is_same_v<T, double>) {
      auto type = H5T_NATIVE_DOUBLE;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    } else {
      static_assert(always_false<T>);
    }
  }

  private:
  static auto checkErrorAndCloseDataset(char const* name, hid_t set, herr_t err) -> void {
    if (err != 0) {
      raisef<std::runtime_error>("dataset read in: {}", name);
    }

    if (H5Dclose(set) != 0) {
      raisef<std::runtime_error>("dataset close in: {}", name);
    }
  }

  hid_t _handle;
};

struct HDF5Writer {
  explicit HDF5Writer(std::filesystem::path const& path)
      : _handle{H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)} { }

  ~HDF5Writer() { H5Fclose(_handle); }

  HDF5Writer(HDF5Writer const& other) = delete;
  HDF5Writer(HDF5Writer&& other)      = delete;

  auto operator=(HDF5Writer const& other) -> HDF5Writer& = delete;
  auto operator=(HDF5Writer&& other) -> HDF5Writer&      = delete;

  auto write(char const* name, stdex::mdspan<double const, stdex::dextents<size_t, 2>> buf) const -> void {
    hsize_t dims[2]{
        static_cast<hsize_t>(buf.extent(0)),
        static_cast<hsize_t>(buf.extent(1)),
    };

    auto def  = H5P_DEFAULT;
    auto type = H5T_NATIVE_DOUBLE;

    auto space  = H5Screate_simple(2, dims, nullptr);
    auto set    = H5Dcreate(_handle, name, type, space, def, def, def);
    auto status = H5Dwrite(set, type, H5S_ALL, H5S_ALL, def, buf.data_handle());

    if (status != 0) {
      raise<std::runtime_error>("error writing dataset\n");
    }

    status = H5Dclose(set);
    if (status != 0) {
      raise<std::runtime_error>("error closing dataset\n");
    }

    status = H5Sclose(space);
    if (status != 0) {
      raise<std::runtime_error>("error closing dataset space\n");
    }
  }

  private:
  hid_t _handle;
};

template<typename T>
[[nodiscard]] auto read(HDF5Reader& reader, char const* dset_str, int ndims, hsize_t* dims) -> std::unique_ptr<T[]> {
  auto dset   = H5Dopen(reader.handle(), dset_str, H5P_DEFAULT);
  auto dspace = H5Dget_space(dset);
  PFFDTD_ASSERT(H5Sget_simple_extent_ndims(dspace) == ndims);

  uint64_t N = 0;
  H5Sget_simple_extent_dims(dspace, dims, nullptr);
  if (ndims == 1) {
    N = dims[0];
  } else if (ndims == 2) {
    N = dims[0] * dims[1];
  } else {
    raisef<std::invalid_argument>("unexpected ndims = {} for dset = {}", ndims, dset_str);
  }

  auto status = herr_t{0};
  auto ptr    = allocate_zeros<T>(N);

  if constexpr (std::is_same_v<T, double>) {
    status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr.get());
  } else if constexpr (std::is_same_v<T, int64_t>) {
    status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr.get());
  } else if constexpr (std::is_same_v<T, int8_t> or std::is_same_v<T, bool>) {
    status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr.get());
  } else {
    static_assert(always_false<T>);
  }

  if (status != 0) {
    raisef<std::runtime_error>("error reading dataset: {}", dset_str);
  }

  if (H5Dclose(dset) != 0) {
    raisef<std::runtime_error>("error closing dataset: {}", dset_str);
  }

  return ptr;
}

} // namespace pffdtd
