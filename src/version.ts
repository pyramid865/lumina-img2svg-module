// Single source of truth for the module identity.
// The bundled UI and the handoff payload MUST report exactly the version
// declared in manifest.json; hard-coding it in multiple places caused v0.1.2
// to ship a package whose ready handshake still announced 0.1.1.
import manifest from "../manifest.json" with { type: "json" };

export const MODULE_ID: string = manifest.id;
export const MODULE_VERSION: string = manifest.version;
