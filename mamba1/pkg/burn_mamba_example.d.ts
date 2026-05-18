/* tslint:disable */
/* eslint-disable */

export function wasm_main(): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly wasm_main: () => any;
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke___wasm_bindgen_2365f5cac11975d9___JsValue__core_e7ca449d3e7a815f___result__Result_____wasm_bindgen_2365f5cac11975d9___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke___web_sys_18b18be49549b6bc___features__gen_IdbVersionChangeEvent__IdbVersionChangeEvent__core_e7ca449d3e7a815f___result__Result_____wasm_bindgen_2365f5cac11975d9___JsValue___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke___js_sys_4f93460bf226ed6f___Function_fn_wasm_bindgen_2365f5cac11975d9___JsValue_____wasm_bindgen_2365f5cac11975d9___sys__Undefined___js_sys_4f93460bf226ed6f___Function_fn_wasm_bindgen_2365f5cac11975d9___JsValue_____wasm_bindgen_2365f5cac11975d9___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke___web_sys_18b18be49549b6bc___features__gen_Event__Event______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures________invoke___web_sys_18b18be49549b6bc___features__gen_Event__Event______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke_______true_: (a: number, b: number) => void;
    readonly wasm_bindgen_2365f5cac11975d9___convert__closures_____invoke_______true__1_: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
