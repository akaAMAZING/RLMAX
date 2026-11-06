//! Minimal CUDA driver API binding: only what RLMAX uses. The runtime needs the driver and nothing else.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuGraph = *mut c_void;
pub(crate) type CuGraphNode = *mut c_void;
type CuGraphExec = *mut c_void;

#[repr(C)]
struct CuKernelNodeParams {
    function: CuFunction,
    grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
    block_x: c_uint, block_y: c_uint, block_z: c_uint,
    shared: c_uint,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
}
pub type DevicePtr = u64;

const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
const ATTR_SM_COUNT: c_int = 16;
const ATTR_MAX_THREADS_PER_SM: c_int = 39;
const ATTR_MAX_REGISTERS_PER_SM: c_int = 82;
const ATTR_MAX_BLOCKS_PER_SM: c_int = 106;

unsafe extern "system" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDriverGetVersion(v: *mut c_int) -> CuResult;
    fn cuDeviceGet(d: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, d: CuDevice) -> CuResult;
    fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, d: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CuContext, d: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease_v2(d: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(ctx: CuContext) -> CuResult;
    fn cuCtxSynchronize() -> CuResult;
    fn cuStreamCreate(s: *mut CuStream, flags: c_uint) -> CuResult;
    fn cuMemcpyHtoDAsync_v2(dst: DevicePtr, src: *const c_void, bytes: usize, s: CuStream) -> CuResult;
    fn cuModuleLoadData(m: *mut CuModule, image: *const c_void) -> CuResult;
    fn cuModuleUnload(m: CuModule) -> CuResult;
    fn cuModuleGetFunction(f: *mut CuFunction, m: CuModule, name: *const c_char) -> CuResult;
    fn cuMemAlloc_v2(p: *mut DevicePtr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(p: DevicePtr) -> CuResult;
    fn cuMemsetD8_v2(p: DevicePtr, value: u8, n: usize) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: DevicePtr, src: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: DevicePtr, bytes: usize) -> CuResult;
    fn cuMemcpyDtoD_v2(dst: DevicePtr, src: DevicePtr, bytes: usize) -> CuResult;
    fn cuLaunchKernel(
        f: CuFunction,
        gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint,
        shared: c_uint, stream: CuStream,
        params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CuResult;
    fn cuGraphCreate(graph: *mut CuGraph, flags: c_uint) -> CuResult;
    fn cuGraphAddKernelNode(node: *mut CuGraphNode, graph: CuGraph, dependencies: *const CuGraphNode,
                            count: usize, params: *const CuKernelNodeParams) -> CuResult;
    fn cuGraphInstantiate_v2(exec: *mut CuGraphExec, graph: CuGraph, error_node: *mut CuGraphNode,
                             log: *mut c_char, log_size: usize) -> CuResult;
    fn cuGraphLaunch(exec: CuGraphExec, stream: CuStream) -> CuResult;
    fn cuGraphDestroy(graph: CuGraph) -> CuResult;
    fn cuGraphExecDestroy(exec: CuGraphExec) -> CuResult;
    fn cuGetErrorString(r: CuResult, s: *mut *const c_char) -> CuResult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CuResult;
    fn cuFuncGetAttribute(v: *mut c_int, attr: c_int, f: CuFunction) -> CuResult;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks: *mut c_int, f: CuFunction, block: c_int, shared: usize) -> CuResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub c_int);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s: *const c_char = std::ptr::null();
        unsafe { cuGetErrorString(self.0, &mut s) };
        let msg = if s.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy() };
        write!(f, "CUDA driver error {}: {msg}", self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(r: CuResult) -> Result<()> {
    if r == 0 { Ok(()) } else { Err(Error(r)) }
}

/// The oldest driver API version the binaries load on: the toolkit's they were built with (13.2 is 13020).
fn min_driver_version() -> c_int {
    let mut parts = option_env!("RLMAX_CUDA_RELEASE").unwrap_or("13.2").split('.').map(|p| p.parse::<c_int>().unwrap_or(0));
    let (major, minor) = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
    major * 1000 + minor * 10
}

pub struct Device {
    dev: CuDevice,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub driver_version: i32,
    pub compute_capability: (i32, i32),
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub max_registers_per_sm: i32,
    pub max_blocks_per_sm: i32,
}

impl Device {
    /// Opens `ordinal`'s primary context. Fails with a clear message on a driver older than the build's toolkit.
    pub fn open(ordinal: i32) -> std::result::Result<Device, String> {
        unsafe {
            check(cuInit(0)).map_err(|e| e.to_string())?;
            let mut v = 0;
            check(cuDriverGetVersion(&mut v)).map_err(|e| e.to_string())?;
            let need = min_driver_version();
            if v < need {
                return Err(format!(
                    "the NVIDIA driver supports CUDA {}.{}; this build needs CUDA {}.{} or newer (update the driver)",
                    v / 1000,
                    (v % 1000) / 10,
                    need / 1000,
                    (need % 1000) / 10
                ));
            }
            let mut dev = 0;
            check(cuDeviceGet(&mut dev, ordinal)).map_err(|e| e.to_string())?;
            let mut ctx = std::ptr::null_mut();
            check(cuDevicePrimaryCtxRetain(&mut ctx, dev)).map_err(|e| e.to_string())?;
            check(cuCtxSetCurrent(ctx)).map_err(|e| e.to_string())?;
            Ok(Device { dev })
        }
    }

    pub fn info(&self) -> Result<DeviceInfo> {
        unsafe {
            let mut buf = [0 as c_char; 256];
            check(cuDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, self.dev))?;
            let attr = |a| {
                let mut v = 0;
                check(cuDeviceGetAttribute(&mut v, a, self.dev)).map(|_| v)
            };
            let mut driver_version = 0;
            check(cuDriverGetVersion(&mut driver_version))?;
            Ok(DeviceInfo {
                name: CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned(),
                driver_version,
                compute_capability: (attr(ATTR_CC_MAJOR)?, attr(ATTR_CC_MINOR)?),
                sm_count: attr(ATTR_SM_COUNT)?,
                max_threads_per_sm: attr(ATTR_MAX_THREADS_PER_SM)?,
                max_registers_per_sm: attr(ATTR_MAX_REGISTERS_PER_SM)?,
                max_blocks_per_sm: attr(ATTR_MAX_BLOCKS_PER_SM)?,
            })
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(cuCtxSynchronize()) }
    }

    /// A non-blocking stream for the `*_raw` launches: work on it can overlap work on other streams on the card.
    /// It lives as long as the context.
    pub fn stream(&self) -> Result<u64> {
        let mut s: CuStream = std::ptr::null_mut();
        unsafe { check(cuStreamCreate(&mut s, 1))? };
        Ok(s as usize as u64)
    }

    /// Free and total device memory in bytes (cuMemGetInfo).
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        unsafe { check(cuMemGetInfo_v2(&mut free, &mut total))? };
        Ok((free, total))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { cuDevicePrimaryCtxRelease_v2(self.dev) };
    }
}

pub struct Module(CuModule);

impl Module {
    /// Loads an embedded fat binary; the driver picks the SASS for the device or JITs the PTX.
    pub fn load(_dev: &Device, image: &[u8]) -> Result<Module> {
        let mut m = std::ptr::null_mut();
        unsafe { check(cuModuleLoadData(&mut m, image.as_ptr().cast()))? };
        Ok(Module(m))
    }

    pub fn function(&self, name: &CStr) -> Result<Function<'_>> {
        let mut f = std::ptr::null_mut();
        unsafe { check(cuModuleGetFunction(&mut f, self.0, name.as_ptr()))? };
        Ok(Function { f, _m: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { cuModuleUnload(self.0) };
    }
}

pub struct Function<'m> {
    f: CuFunction,
    _m: &'m Module,
}

impl Function<'_> {
    /// Local memory per thread in bytes (CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES): the kernel's frame for this device.
    pub fn local_bytes(&self) -> Result<i32> {
        let mut v = 0;
        unsafe { check(cuFuncGetAttribute(&mut v, 3, self.f))? };
        Ok(v)
    }

    /// Registers per thread (CU_FUNC_ATTRIBUTE_NUM_REGS).
    pub fn num_regs(&self) -> Result<i32> {
        let mut v = 0;
        unsafe { check(cuFuncGetAttribute(&mut v, 4, self.f))? };
        Ok(v)
    }

    /// Blocks of `block` threads resident per SM, as the driver computes it (registers, local memory, shared memory).
    pub fn max_active_blocks(&self, block: u32) -> Result<i32> {
        let mut v = 0;
        unsafe { check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut v, self.f, block as c_int, 0))? };
        Ok(v)
    }

    /// Launches on the default stream. `params` are pointers to each kernel argument, in order.
    ///
    /// # Safety
    /// The argument list must match the kernel's signature exactly.
    pub unsafe fn launch(&self, grid: u32, block: u32, params: &mut [*mut c_void]) -> Result<()> {
        unsafe { self.launch_on(grid, block, params, 0) }
    }

    /// As [`Function::launch`] on a stream given by its raw handle (0: the legacy default stream, which PyTorch's
    /// default stream is too; `torch.cuda.current_stream().cuda_stream` otherwise).
    ///
    /// # Safety
    /// The argument list must match the kernel's signature exactly, every pointer in it valid for the launch, and
    /// `stream` a live stream of this context (or 0).
    pub unsafe fn launch_on(&self, grid: u32, block: u32, params: &mut [*mut c_void], stream: u64) -> Result<()> {
        unsafe {
            check(cuLaunchKernel(
                self.f, grid, 1, 1, block, 1, 1, 0, stream as usize as CuStream,
                params.as_mut_ptr(), std::ptr::null_mut(),
            ))
        }
    }
}

// Graph nodes copy their argument values at insertion. The owning batch must
// keep the module, buffers and every device pointer in those values alive.
pub(crate) struct GraphBuilder {
    graph: CuGraph,
}
impl GraphBuilder {
    pub(crate) fn new() -> Result<Self> {
        let mut graph = std::ptr::null_mut();
        unsafe { check(cuGraphCreate(&mut graph, 0))?; }
        Ok(Self { graph })
    }

    // Graph node: callers supply the full set of predecessor nodes.
    pub(crate) unsafe fn add_after(&mut self, function: &Function<'_>, grid: u32, block: u32,
        params: &mut [*mut c_void], dependencies: &[CuGraphNode]) -> Result<CuGraphNode> {
        let node_params = CuKernelNodeParams { function: function.f, grid_x: grid, grid_y: 1, grid_z: 1,
            block_x: block, block_y: 1, block_z: 1, shared: 0, params: params.as_mut_ptr(), extra: std::ptr::null_mut() };
        let mut node=std::ptr::null_mut();
        let deps=if dependencies.is_empty() { std::ptr::null() } else { dependencies.as_ptr() };
        unsafe { check(cuGraphAddKernelNode(&mut node,self.graph,deps,dependencies.len(),&node_params))?; }
        Ok(node)
    }

    pub(crate) fn instantiate(&self) -> Result<GraphExec> {
        let mut exec = std::ptr::null_mut();
        unsafe { check(cuGraphInstantiate_v2(&mut exec, self.graph, std::ptr::null_mut(), std::ptr::null_mut(), 0))?; }
        Ok(GraphExec(exec))
    }
}
impl Drop for GraphBuilder {
    fn drop(&mut self) { unsafe { cuGraphDestroy(self.graph); } }
}
pub(crate) struct GraphExec(CuGraphExec);
impl GraphExec {
    pub(crate) unsafe fn launch(&self, stream: u64) -> Result<()> {
        unsafe { check(cuGraphLaunch(self.0, stream as usize as CuStream)) }
    }
}
impl Drop for GraphExec {
    fn drop(&mut self) { unsafe { cuGraphExecDestroy(self.0); } }
}

/// A device allocation of `len` elements of `T`.
pub struct Buffer<T: Copy> {
    ptr: DevicePtr,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

impl<T: Copy> Buffer<T> {
    pub fn from_slice(_dev: &Device, data: &[T]) -> Result<Buffer<T>> {
        let bytes = std::mem::size_of_val(data);
        let mut ptr = 0;
        unsafe {
            check(cuMemAlloc_v2(&mut ptr, bytes.max(1)))?;
            check(cuMemcpyHtoD_v2(ptr, data.as_ptr().cast(), bytes))?;
        }
        Ok(Buffer { ptr, len: data.len(), _t: std::marker::PhantomData })
    }

    /// `len` zeroed elements, cleared on the device: no host array the size of the allocation (`T` must be valid as all zero bits).
    pub fn zeroed(_dev: &Device, len: usize) -> Result<Buffer<T>> {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).expect("buffer size overflow");
        let mut ptr = 0;
        unsafe {
            check(cuMemAlloc_v2(&mut ptr, bytes.max(1)))?;
            check(cuMemsetD8_v2(ptr, 0, bytes))?;
        }
        Ok(Buffer { ptr, len, _t: std::marker::PhantomData })
    }

    /// A new allocation holding a device-side copy of `src` (ordered after the kernels already launched).
    pub fn copy_of(_dev: &Device, src: &Buffer<T>) -> Result<Buffer<T>> {
        let bytes = src.len * std::mem::size_of::<T>();
        let mut ptr = 0;
        unsafe { check(cuMemAlloc_v2(&mut ptr, bytes.max(1)))? };
        let out = Buffer { ptr, len: src.len, _t: std::marker::PhantomData };
        unsafe { check(cuMemcpyDtoD_v2(out.ptr, src.ptr, bytes))? };
        Ok(out)
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        assert_eq!(data.len(), self.len);
        unsafe { check(cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), std::mem::size_of_val(data))) }
    }

    /// Queues the write on `stream`, in order with that stream's launches.
    ///
    /// # Safety
    /// `data` must stay alive and unchanged until the stream has run the copy.
    pub unsafe fn write_on(&self, data: &[T], stream: u64) -> Result<()> {
        assert_eq!(data.len(), self.len);
        unsafe { check(cuMemcpyHtoDAsync_v2(self.ptr, data.as_ptr().cast(), std::mem::size_of_val(data), stream as usize as CuStream)) }
    }

    pub fn read(&self, out: &mut [T]) -> Result<()> {
        assert_eq!(out.len(), self.len);
        unsafe { check(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, std::mem::size_of_val(out))) }
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.ptr) };
    }
}
