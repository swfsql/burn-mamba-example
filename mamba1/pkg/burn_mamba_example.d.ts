/* tslint:disable */
/* eslint-disable */

export function wasm_main(): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly wasm_main: () => any;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___closure__destroy___dyn_core_dde6c4b55a98adc4___ops__function__FnMut__wasm_bindgen_7def5f3d4cd1d6ed___JsValue____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___closure__destroy___dyn_core_dde6c4b55a98adc4___ops__function__FnMut__web_sys_76b9bbb79fb22d11___features__gen_IdbVersionChangeEvent__IdbVersionChangeEvent____Output___core_dde6c4b55a98adc4___result__Result_____wasm_bindgen_7def5f3d4cd1d6ed___JsValue___: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___closure__destroy___dyn_for__a__core_dde6c4b55a98adc4___ops__function__Fn____a_web_sys_76b9bbb79fb22d11___features__gen_Event__Event____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___closure__destroy___dyn_core_dde6c4b55a98adc4___ops__function__FnMut_____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___closure__destroy___dyn_core_dde6c4b55a98adc4___ops__function__Fn__web_sys_76b9bbb79fb22d11___features__gen_Event__Event____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke___web_sys_76b9bbb79fb22d11___features__gen_IdbVersionChangeEvent__IdbVersionChangeEvent__core_dde6c4b55a98adc4___result__Result_____wasm_bindgen_7def5f3d4cd1d6ed___JsValue__: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke___wasm_bindgen_7def5f3d4cd1d6ed___JsValue__wasm_bindgen_7def5f3d4cd1d6ed___JsValue_____: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke___wasm_bindgen_7def5f3d4cd1d6ed___JsValue_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures________invoke___web_sys_76b9bbb79fb22d11___features__gen_Event__Event_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke___web_sys_76b9bbb79fb22d11___features__gen_Event__Event_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke______: (a: number, b: number) => void;
    readonly wasm_bindgen_7def5f3d4cd1d6ed___convert__closures_____invoke_______1_: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
