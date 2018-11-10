#pragma once

#include <cstddef>

/**
 * Python doesn't have a pointer type, therefore we create a pointer wrapper
 * see https://stackoverflow.com/questions/48982143/returning-and-passing-around-raw-pod-pointers-arrays-with-python-c-and-pyb?rq=1
 */
template <typename T>
class ptr {
public:
    ptr() : p(nullptr) {}
    ptr(T* p) : p(p) {}
    ptr(std::size_t p) : p((T*)p) {}
    ptr(const ptr& other) : ptr(other.p) {}
    T& operator* () const { return *p; }
    T* operator->() const { return  p; }
    T* get() const { return p; }
    void destroy() { delete p; }
    T& operator[](std::size_t idx) const { return p[idx]; }
    bool is_null() const { return p == nullptr; }
private:
    T* p;
};
