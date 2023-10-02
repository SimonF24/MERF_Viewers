import * as React from 'react';
import * as THREE from 'three';
import Box from '@mui/material/Box';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import IconButton from '@mui/material/IconButton';

import CircularProgressWithLabel from
    '@/components/circular_progress_with_label';
import {
    createNetworkWeightTexture, createViewDependenceFunctions
} from 'viewdependency.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';

import rayMarchFragmentShaderBodyRaw from
    'fragment.glsl';
import viewDependenceNetworkShaderFunctions from
    'viewdependency.glsl';

const rayMarchFragmentShaderBody = rayMarchFragmentShaderBodyRaw.slice(17);

/** @const {string} */
const accumulateFragmentShader = `
varying vec2 vUv;
uniform vec2 lowResolution;
uniform vec2 highResolution;
uniform vec2 jitterOffset;
uniform float emaAlpha;

uniform sampler2D mapLowRes;
uniform sampler2D mapHistory;

float pixelFilter(vec2 pixelCenter, vec2 sampleCenter) {
  vec2 delta = pixelCenter - sampleCenter;
  float squaredNorm = dot(delta, delta);
  return exp(-2.29 * squaredNorm);
}

void main() {
  // First we need to compute the coordinates of the pixel centers
  // in the low resolution grid by compensating for the camera jitter.
  // Note that the offset is defined in clip space [-1,1]^2, so we need
  // to multiply it by 0.5 to make it valid in texture space [0,1]^2.
  vec2 compensatedUnitCoords = vUv - jitterOffset * 0.5;

  // Now compute the integer coordinates in the low resolution grid for each
  // adjacent texel.
  ivec2 lowResCoords00 = ivec2(compensatedUnitCoords * lowResolution - 0.5);
  ivec2 lowResCoords01 = ivec2(0, 1) + lowResCoords00;
  ivec2 lowResCoords10 = ivec2(1, 0) + lowResCoords00;
  ivec2 lowResCoords11 = ivec2(1, 1) + lowResCoords00;

  float mask00 =
    min(lowResCoords00.x, lowResCoords00.y) < 0 ||
    lowResCoords00.x >= int(lowResolution.x) ||
    lowResCoords00.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask01 =
    min(lowResCoords01.x, lowResCoords01.y) < 0 ||
    lowResCoords01.x >= int(lowResolution.x) ||
    lowResCoords01.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask10 =
    min(lowResCoords10.x, lowResCoords10.y) < 0 ||
    lowResCoords10.x >= int(lowResolution.x) ||
    lowResCoords10.y >= int(lowResolution.y) ? 0.0 : 1.0;
  float mask11 =
    min(lowResCoords11.x, lowResCoords11.y) < 0 ||
    lowResCoords11.x >= int(lowResolution.x) ||
    lowResCoords11.y >= int(lowResolution.y) ? 0.0 : 1.0;

  // We also need to keep track of the high resolution counterparts of these
  // coordinates, so we can compute the pixel reconstruction filter weights.
  vec2 compensatedHighResCoords = highResolution * compensatedUnitCoords;
  vec2 highResCoords00 =
      highResolution * (vec2(lowResCoords00) + 0.5) / lowResolution;
  vec2 highResCoords01 =
      highResolution * (vec2(lowResCoords01) + 0.5) / lowResolution;
  vec2 highResCoords10 =
      highResolution * (vec2(lowResCoords10) + 0.5) / lowResolution;
  vec2 highResCoords11 =
      highResolution * (vec2(lowResCoords11) + 0.5) / lowResolution;

  vec4 lowResColor = vec4(0.0, 0.0, 0.0, 0.0);
  lowResColor += mask00 * vec4(
    texelFetch(mapLowRes,lowResCoords00, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords00);
  lowResColor += mask01 * vec4(
    texelFetch(mapLowRes, lowResCoords01, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords01);
  lowResColor += mask10 * vec4(
    texelFetch(mapLowRes, lowResCoords10, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords10);
  lowResColor += mask11 * vec4(
    texelFetch(mapLowRes, lowResCoords11, 0).rgb, 1.0) *
    pixelFilter(compensatedHighResCoords, highResCoords11);

  vec4 historyColor = texture2D(mapHistory, vUv);
  gl_FragColor = emaAlpha * lowResColor + (1.0 - emaAlpha) * historyColor;
}
`;

/** @const {string}  */
const accumulateVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/** @const {string} */
const normalizeFragmentShader = `
varying vec2 vUv;
uniform sampler2D map;
void main() {
  gl_FragColor = texture2D(map, vUv);
  if (gl_FragColor.a > 0.0) {
    gl_FragColor.rgb /= gl_FragColor.a;
  }
  gl_FragColor.a = 1.0;
}
`;

/** @const {string}  */
const normalizeVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/**
 * The vertex shader for rendering a baked MERF scene with ray marching.
 * @const {string}
 */
const rayMarchVertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
uniform mat4 world_T_clip;

void main() {
vec4 posClip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
gl_Position = posClip;

posClip /= posClip.w;
vec4 nearPoint = world_T_clip * vec4(posClip.x, posClip.y, -1.0, 1.0);
vec4 farPoint = world_T_clip * vec4(posClip.x, posClip.y, 1.0, 1.0);

vOrigin = nearPoint.xyz / nearPoint.w;
vDirection = normalize(farPoint.xyz / farPoint.w - vOrigin);
}
`;

/**
 * We build the ray marching shader programmatically, this string contains the
 * header for the shader.
 * @const {string}
 */
const rayMarchFragmentShaderHeader = `
precision highp float;

varying vec3 vOrigin;
varying vec3 vDirection;

uniform int displayMode;

uniform vec3 minPosition;
uniform mat3 worldspaceROpengl;
uniform float nearPlane;

uniform highp sampler3D occupancyGrid_L0;
uniform highp sampler3D occupancyGrid_L1;
uniform highp sampler3D occupancyGrid_L2;
uniform highp sampler3D occupancyGrid_L3;
uniform highp sampler3D occupancyGrid_L4;

uniform float voxelSizeOccupancy_L0;
uniform float voxelSizeOccupancy_L1;
uniform float voxelSizeOccupancy_L2;
uniform float voxelSizeOccupancy_L3;
uniform float voxelSizeOccupancy_L4;

uniform vec3 gridSizeOccupancy_L0;
uniform vec3 gridSizeOccupancy_L1;
uniform vec3 gridSizeOccupancy_L2;
uniform vec3 gridSizeOccupancy_L3;
uniform vec3 gridSizeOccupancy_L4;

uniform highp sampler2D weightsZero;
uniform highp sampler2D weightsOne;
uniform highp sampler2D weightsTwo;

uniform int stepMult;
uniform float rangeFeaturesMin;
uniform float rangeFeaturesMax;
uniform float rangeDensityMin;
uniform float rangeDensityMax;

#ifdef USE_SPARSE_GRID
uniform vec3 sparseGridGridSize;
uniform float sparseGridVoxelSize;
uniform vec3 atlasSize;
uniform float dataBlockSize;
uniform highp sampler3D sparseGridRgb;
uniform highp sampler3D sparseGridDensity;
uniform highp sampler3D sparseGridFeatures;
uniform highp sampler3D sparseGridBlockIndices;
#endif

#ifdef USE_TRIPLANE
uniform vec2 triplaneGridSize;
uniform float triplaneVoxelSize;
// need to use texture arrays, otherwise we exceed max texture unit limit
uniform highp sampler2DArray planeRgb;
uniform highp sampler2DArray planeDensity;
uniform highp sampler2DArray planeFeatures;
#endif
`;

/**
 * @param {!THREE.Texture} textureLowRes
 * @param {!THREE.Texture} textureHistory
 * @param {!THREE.Vector2} lowResolution
 * @param {!THREE.Vector2} highResolution
 * @return {!THREE.Material}
 */
function createAccumulateMaterial(
    textureLowRes, textureHistory, lowResolution, highResolution) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      'mapLowRes': {'value': textureLowRes},
      'mapHistory': {'value': textureHistory},
      'lowResolution': {'value': lowResolution},
      'highResolution': {'value': highResolution},
      'jitterOffset': {'value': new THREE.Vector2(0.0, 0.0)},
      'emaAlpha': {'value': 0.15},
    },
    vertexShader: accumulateVertexShader,
    fragmentShader: accumulateFragmentShader,
  });

  return material;
}

/**
 * @param {!THREE.Texture} texture
 * @return {!THREE.Material}
 */
function createNormalizeMaterial(texture) {
    const material = new THREE.ShaderMaterial({
      uniforms: {
        'map': {'value': texture},
      },
      vertexShader: normalizeVertexShader,
      fragmentShader: normalizeFragmentShader,
    });
  
    return material;
  }

/**
 * Creates three equally sized textures holding triplanes.
 * @param {Uint8Array} data Texture data array
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} format Format of the texture
 * @return {!THREE.DataArrayTexture} Texture array of size three
 */
function createTriplaneTextureArray(data, width, height, format) {
    let texture = new THREE.DataArrayTexture(data, width, height, 3);
    texture.format = format;
    texture.generateMipmaps = false;
    texture.magFilter = texture.minFilter = THREE.LinearFilter;
    texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
    texture.type = THREE.UnsignedByteType;
    texture.needsUpdate = true;
    return texture;
}

/**
 * Creates a volume texture.
 * @param {number} width Width of the texture
 * @param {number} height Height of the texture
 * @param {number} depth Depth of the texture
 * @param {number} format Format of the texture
 * @param {number} filter Filter strategy of the texture
 * @return {!THREE.Data3DTexture} Volume texture
 */
function createVolumeTexture(data, width, height, depth,
    format, filter) {
    let volumeTexture = new THREE.Data3DTexture(
        data, width, height, depth);
    volumeTexture.format = format;
    volumeTexture.generateMipmaps = false;
    volumeTexture.magFilter = volumeTexture.minFilter = filter;
    volumeTexture.wrapS = volumeTexture.wrapT = volumeTexture.wrapR =
        THREE.ClampToEdgeWrapping;
    volumeTexture.type = THREE.UnsignedByteType;
    volumeTexture.needsUpdate = true;
    return volumeTexture;
}

/**
 * Extends a dictionary.
 * @param {!object} obj Dictionary to extend
 * @param {!object} src Dictionary to be written into obj
 * @return {!object} Extended dictionary
 */
function extend(obj, src) {
    for (let key in src) {
        if (src.hasOwnProperty(key)) obj[key] = src[key];
    }
    return obj;
}

/**
 * Checks if an object is empty
 * @param {!object} obj 
 * @returns {!boolean}
 */
function isEmpty(obj) {
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) return false;
    }
    return true;
}

export default function MERF(props) {

    if (props.occupancyGrids.length == 0
        || props.planeData.length == 0
        || isEmpty(props.sceneParams)
        || props.sparseGridBlockIndices == null
        || props.sparseGridFeatures.length == 0
        || props.sparseGridRgbAndDensity.length == 0) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    height: '100%'
                }}
            >
                <Box
                    sx={{
                        margin: 'auto',
                    }}
                >
                    <CircularProgressWithLabel
                        sx={{
                            margin: 'auto'
                        }}
                        value={props.loadingProgress*100}
                    />
                </Box>
            </Box>
        )
    }

    const canvasRef = React.useRef();
    const [currentlyFullScreen, setCurrentlyFullScreen] = React.useState(false);
    const viewspaceRef = React.useRef();
    const viewspaceContainerRef = React.useRef();

    function toggleFullscreen() {
        if (document.fullscreenElement) {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.mozCancelFullScreen) { // Firefox
                document.mozCancelFullScreen();
            } else if (document.webkitExitFullscreen) { // Chrome, Safari and Opera
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) { // IE/Edge
                document.msExitFullscreen();
            }
        } else {
            const viewspace = viewspaceRef.current;
            if (viewspace.requestFullscreen) {
                viewspace.requestFullscreen();
            } else if (viewspace.mozRequestFullScreen) { // Firefox
                viewspace.mozRequestFullScreen();
            } else if (viewspace.webkitRequestFullscreen) { // Chrome, Safari and Opera
                viewspace.webkitRequestFullscreen();
            } else if (viewspace.msRequestFullscreen) { // IE/Edge
                viewspace.msRequestFullscreen();
            }
        }
    }

    React.useEffect(() => {

        if (props.loadingProgress < 1) {
            return
        }

        const sceneParams = props.sceneParams;

        if (props.showStats) {
            /**
             * Our framerate display.
             * @type {?Object}
             */
            var gStats = null;
        }

        /**
         * Different display modes for debugging rendering.
         * @enum {number}
         */
        const DisplayModeType = {
        /** Runs the full model with view dependence. */
        DISPLAY_NORMAL: 0,
        /** Disables the view-dependence network. */
        DISPLAY_DIFFUSE: 1,
        /** Only shows the latent features. */
        DISPLAY_FEATURES: 2,
        /** Only shows the view dependent component. */
        DISPLAY_VIEW_DEPENDENT: 3,
        /** Only shows the coarse block grid. */
        DISPLAY_COARSE_GRID: 4,
        };

        /**  @type {!DisplayModeType}  */
        let gDisplayMode = DisplayModeType.DISPLAY_NORMAL;

        /**
         * Number of sample points per voxel.
         * @type {number}
         */
        let gStepMult = 1;

        /**
         * We control the camera using either orbit controls...
         * @type {?OrbitControls}
         */
        let gOrbitControls = null;

        /**
         * This is the main bookkeeping scene for progressive upsampling, as it
         * keeps track of multiple low-res frames, and their corresponding filter
         * weights.
         * @type {?THREE.Scene}
         */
        let gSceneAccumulate = null;

        /**
         * A lower res orthographic camera used to kick off ray marching
         * with a full-screen render pass.
         * @type {?THREE.OrthographicCamera}
         */
        let gLowResBlitCamera = null;

        /**
         * A higher res orthographic camera used to perform full-resolution
         * post-processing passes.
         * @type {?THREE.OrthographicCamera}
         */
        let gHighResBlitCamera = null;

        /**
         * Counts the current frame number, used for random sampling.
         * @type {number}
         */
        let gFrameIndex = 0;

        /**
         * This is a half-res rendertarget used for progressive rendering.
         * @type {?THREE.WebGLRenderTarget}
         */
        let gLowResTexture = null;

        /**
         * These are the ping-pong buffers used for progressive upsampling. Every frame
         * we read from one buffer, and write into the other. This allows us to maintain
         * a history of multiple low-res frames, and their corresponding filter
         * weights.
         * @type {!Array<?THREE.WebGLRenderTarget>}
         */
        const gAccumulationTextures = [null, null];

        /**
         * Blits a texture into the framebuffer, normalizing the result using
         * the alpha channel. I.e. pixel_out = pixel_in.rgba / pixel_in.a.
         * @type {?THREE.Scene}
         */
        let gSceneNormalize = null;

        /**
         * Loads the MERF scene.
         * @param {number} width Width of the viewer frame
         * @param {number} height Height of the viewer frame
         * @param {!bool} useLargerStepsWhenOccluded Enables a hack that speeds up
         * rendering by using a larger step size with decreasing visibility
         * @param {number} nearPlane Distance to the near clipping plane
         */
        function loadScene(
            width, height, useLargerStepsWhenOccluded,
            nearPlane, gCamera, gOrbitControls,
            gRenderer) {

            const useSparseGrid = sceneParams['sparse_grid_resolution'] > 0;

            const useTriplane = sceneParams['triplane_resolution'] > 0;

            // Load sparse grid, note that textures are only allocated and
            // the actual loading is done progressively in loadOnFirstFrame.
            let sparseGridRgbTexture = null;
            let sparseGridDensityTexture = null;
            let sparseGridFeaturesTexture = null;
            let sparseGridBlockIndicesTexture = null;
            if (useSparseGrid) {

                let sparseGridDensity = new Uint8Array(
                    sceneParams['atlas_width'] * sceneParams['atlas_height']
                    * sceneParams['atlas_depth']
                );
                let sparseGridFeatures = new Uint8Array(
                    sceneParams['atlas_width'] * sceneParams['atlas_height']
                    * sceneParams['atlas_depth'] * 4
                );
                let sparseGridRgb = new Uint8Array(
                    sceneParams['atlas_width'] * sceneParams['atlas_height']
                    * sceneParams['atlas_depth'] * 4
                );

                let densityOffset = 0;
                let offset = 0;
                for (let i = 0; i < sceneParams['num_slices']; i++) {
                    const rgbaImage = props.sparseGridRgbAndDensity[i];
                    // The png's RGB channels hold RGB and the png's alpha
                    // channel holds density. We split apart RGB and density
                    // and upload to two distinct textures, so we can
                    // separately query these quantities.
                    let rgbPixels = new Uint8Array(
                        sceneParams['atlas_width']
                        * sceneParams['atlas_height'] * 4
                    );
                    let densityPixels = new Uint8Array(
                        sceneParams['atlas_width']
                        * sceneParams['atlas_height']
                    );
                    for (let j = 0;
                        j < sceneParams['atlas_width']
                        * sceneParams['atlas_height']; j++) {
                        rgbPixels[j * 4 + 0] = rgbaImage[j * 4 + 0];
                        rgbPixels[j * 4 + 1] = rgbaImage[j * 4 + 1];
                        rgbPixels[j * 4 + 2] = rgbaImage[j * 4 + 2];
                        rgbPixels[j * 4 + 3] = 1;
                        densityPixels[j] = rgbaImage[j * 4 + 3];
                    }
                    sparseGridDensity.set(densityPixels, densityOffset);
                    densityOffset += sceneParams['atlas_width']
                        * sceneParams['atlas_height'];
                    sparseGridFeatures.set(
                        props.sparseGridFeatures[i], offset
                    );
                    sparseGridRgb.set(rgbPixels, offset);
                    offset += sceneParams['atlas_width']
                        * sceneParams['atlas_height'] * 4;
                }

                sparseGridDensityTexture = createVolumeTexture(
                    sparseGridDensity, sceneParams['atlas_width'],
                    sceneParams['atlas_height'], sceneParams['atlas_depth'],
                    THREE.RedFormat, THREE.LinearFilter
                );
                sparseGridFeaturesTexture = createVolumeTexture(
                    sparseGridFeatures, sceneParams['atlas_width'],
                    sceneParams['atlas_height'], sceneParams['atlas_depth'],
                    THREE.RGBAFormat, THREE.LinearFilter
                );
                sparseGridRgbTexture = createVolumeTexture(
                    sparseGridRgb, sceneParams['atlas_width'],
                    sceneParams['atlas_height'], sceneParams['atlas_depth'],
                    THREE.RGBAFormat, THREE.LinearFilter
                );

                // The indirection grid uses nearest filtering and is loaded
                // in one go.
                let v = sceneParams['sparse_grid_resolution'] /
                    sceneParams['data_block_size'];

                const sparseGridBlockIndicesImage =
                    props.sparseGridBlockIndices;
                // The indirection grid is stored as a single channel image,
                // so we need to expand it to RGBA.
                let sparseGridBlockIndicesRgba = new Uint8Array(v * v * v * 4);
                for (let i = 0; i < v*v*v; i++) {
                    sparseGridBlockIndicesRgba[i * 4 + 0] =
                        sparseGridBlockIndicesImage[i * 3 + 0];
                    sparseGridBlockIndicesRgba[i * 4 + 1] =
                        sparseGridBlockIndicesImage[i * 3 + 1];
                    sparseGridBlockIndicesRgba[i * 4 + 2] =
                        sparseGridBlockIndicesImage[i * 3 + 2];
                    sparseGridBlockIndicesRgba[i * 4 + 3] = 1;
                }
                sparseGridBlockIndicesTexture = createVolumeTexture(
                    sparseGridBlockIndicesRgba, v, v, v, THREE.RGBAFormat,
                    THREE.NearestFilter
                );
            }

            // Load triplanes.
            let planeRgbTexture, planeDensityTexture, planeFeaturesTexture,
                triplaneGridSize;
            if (useTriplane) {
                let triplaneResolution = sceneParams['triplane_resolution'];
                let triplaneNumTexels = triplaneResolution * triplaneResolution;
                triplaneGridSize =
                    new THREE.Vector2(triplaneResolution, triplaneResolution);
                const planeData = props.planeData;
                let planeRgbStack = new Uint8Array(3 * triplaneNumTexels * 4);
                let planeDensityStack = new Uint8Array(3 * triplaneNumTexels);
                let planeFeaturesStack =
                    new Uint8Array(3 * triplaneNumTexels * 4);
                for (let plane_idx = 0; plane_idx < 3; plane_idx++) {
                    let baseOffset = plane_idx * triplaneNumTexels;
                    let planeRgbAndDensity = planeData[2 * plane_idx];
                    let planeFeatures = planeData[2 * plane_idx + 1];
                    for (let j = 0; j < triplaneNumTexels; j++) {
                    planeRgbStack[(baseOffset + j) * 4 + 0] =
                        planeRgbAndDensity[j * 4 + 0];
                    planeRgbStack[(baseOffset + j) * 4 + 1] =
                        planeRgbAndDensity[j * 4 + 1];
                    planeRgbStack[(baseOffset + j) * 4 + 2] =
                        planeRgbAndDensity[j * 4 + 2];
                    planeRgbStack[(baseOffset + j) * 4 + 3] = 1;
                    planeDensityStack[baseOffset + j] =
                        planeRgbAndDensity[j * 4 + 3];
                    planeFeaturesStack[(baseOffset + j) * 4 + 0] =
                        planeFeatures[j * 4 + 0];
                    planeFeaturesStack[(baseOffset + j) * 4 + 1] =
                        planeFeatures[j * 4 + 1];
                    planeFeaturesStack[(baseOffset + j) * 4 + 2] =
                        planeFeatures[j * 4 + 2];
                    planeFeaturesStack[(baseOffset + j) * 4 + 3] =
                        planeFeatures[j * 4 + 3];
                    }
                }
                planeRgbTexture = createTriplaneTextureArray(
                    planeRgbStack, triplaneResolution, triplaneResolution,
                    THREE.RGBAFormat
                );
                planeDensityTexture = createTriplaneTextureArray(
                    planeDensityStack, triplaneResolution, triplaneResolution,
                    THREE.RedFormat
                );
                planeFeaturesTexture = createTriplaneTextureArray(
                    planeFeaturesStack, triplaneResolution, triplaneResolution,
                    THREE.RGBAFormat
                );
            }

            // Load occupancy grids for empty space skipping.
            let resolutionToUse, voxelSizeToUse;
            if (useTriplane) {
                resolutionToUse = sceneParams['triplane_resolution'];
                voxelSizeToUse = sceneParams['triplane_voxel_size'];
            } else {
                resolutionToUse = sceneParams['sparse_grid_resolution'];
                voxelSizeToUse = sceneParams['sparse_grid_voxel_size'];
            }
            
            const occupancyGridBlockSizes = [8, 16, 32, 64, 128];
            let occupancyGridTextures = [];
            let occupancyGridSizes = [];
            let occupancyVoxelSizes = [];
            for (let occupancyGridIndex = 0;
                    occupancyGridIndex < occupancyGridBlockSizes.length;
                    occupancyGridIndex++) {
                let occupancyGridImageFourChannels =
                    props.occupancyGrids[occupancyGridIndex]
                const occupancyGridBlockSize =
                    occupancyGridBlockSizes[occupancyGridIndex];
                // Assuming width = height = depth which typically holds 
                // when employing scene contraction.
                const v = Math.ceil(resolutionToUse / occupancyGridBlockSize);
                occupancyGridSizes.push(new THREE.Vector3(v, v, v));
                occupancyVoxelSizes.push(
                    voxelSizeToUse * occupancyGridBlockSize
                );
                let occupancyGridImage = new Uint8Array(v * v * v);
                for (let j = 0; j < v * v * v; j++) {
                    occupancyGridImage[j] = 
                        occupancyGridImageFourChannels[4 * j];
                }
                const occupancyGridTexture = createVolumeTexture(
                    occupancyGridImage, v, v, v, THREE.RedFormat,
                    THREE.NearestFilter
                )
                occupancyGridTextures.push(occupancyGridTexture);
            }

            // Assemble shader code from header, on-the-fly generated view-dependency
            // functions and body.
            let fragmentShaderSource = rayMarchFragmentShaderHeader;
            fragmentShaderSource += createViewDependenceFunctions(
                sceneParams, viewDependenceNetworkShaderFunctions);
            fragmentShaderSource += rayMarchFragmentShaderBody;

            // Upload networks weights into textures (biases are written as
            // compile-time constants into the shader).
            let weightsTexZero = createNetworkWeightTexture(sceneParams['0_weights']);
            let weightsTexOne = createNetworkWeightTexture(sceneParams['1_weights']);
            let weightsTexTwo = createNetworkWeightTexture(sceneParams['2_weights']);

            let worldspaceROpengl = new THREE.Matrix3();
            worldspaceROpengl.set(-1, 0, 0, 0, 0, 1, 0, 1, 0);
            let minPosition = new THREE.Vector3(-2.0, -2.0, -2.0);

            // Pass uniforms to the shader.
            let rayMarchUniforms = {
                'occupancyGrid_L4': {'value': occupancyGridTextures[0]},
                'occupancyGrid_L3': {'value': occupancyGridTextures[1]},
                'occupancyGrid_L2': {'value': occupancyGridTextures[2]},
                'occupancyGrid_L1': {'value': occupancyGridTextures[3]},
                'occupancyGrid_L0': {'value': occupancyGridTextures[4]},
                'voxelSizeOccupancy_L4': {'value': occupancyVoxelSizes[0]},
                'voxelSizeOccupancy_L3': {'value': occupancyVoxelSizes[1]},
                'voxelSizeOccupancy_L2': {'value': occupancyVoxelSizes[2]},
                'voxelSizeOccupancy_L1': {'value': occupancyVoxelSizes[3]},
                'voxelSizeOccupancy_L0': {'value': occupancyVoxelSizes[4]},
                'gridSizeOccupancy_L4': {'value': occupancyGridSizes[0]},
                'gridSizeOccupancy_L3': {'value': occupancyGridSizes[1]},
                'gridSizeOccupancy_L2': {'value': occupancyGridSizes[2]},
                'gridSizeOccupancy_L1': {'value': occupancyGridSizes[3]},
                'gridSizeOccupancy_L0': {'value': occupancyGridSizes[4]},

                'displayMode': {'value': gDisplayMode - 0},
                'nearPlane': {'value': nearPlane},
                'minPosition': {'value': minPosition},
                'weightsZero': {'value': weightsTexZero},
                'weightsOne': {'value': weightsTexOne},
                'weightsTwo': {'value': weightsTexTwo},
                'world_T_clip': {'value': new THREE.Matrix4()},
                'worldspaceROpengl': {'value': worldspaceROpengl},

                'stepMult': {'value': gStepMult},
                'rangeFeaturesMin': {'value': sceneParams['range_features'][0]},
                'rangeFeaturesMax': {'value': sceneParams['range_features'][1]},
                'rangeDensityMin': {'value': sceneParams['range_density'][0]},
                'rangeDensityMax': {'value': sceneParams['range_density'][1]},
            };

            if (useTriplane) {
                let triplaneUniforms = {
                'planeRgb': {'value': planeRgbTexture},
                'planeDensity': {'value': planeDensityTexture},
                'planeFeatures': {'value': planeFeaturesTexture},
                'triplaneGridSize': {'value': triplaneGridSize},
                'triplaneVoxelSize': {'value': sceneParams['triplane_voxel_size']},
                };
                rayMarchUniforms = extend(rayMarchUniforms, triplaneUniforms);
                fragmentShaderSource = '#define USE_TRIPLANE\n' + fragmentShaderSource;
            }
            if (useSparseGrid) {
                let sparseGridUniforms = {
                'sparseGridRgb': {'value': sparseGridRgbTexture},
                'sparseGridDensity': {'value': sparseGridDensityTexture},
                'sparseGridFeatures': {'value': sparseGridFeaturesTexture},
                'sparseGridBlockIndices': {'value': sparseGridBlockIndicesTexture},
                'dataBlockSize': {'value': sceneParams['data_block_size']},
                'sparseGridVoxelSize':
                    {'value': sceneParams['sparse_grid_voxel_size']},
                'sparseGridGridSize': {
                    'value': new THREE.Vector3(
                        sceneParams['sparse_grid_resolution'],
                        sceneParams['sparse_grid_resolution'],
                        sceneParams['sparse_grid_resolution'])
                },
                'atlasSize': {
                    'value': new THREE.Vector3(
                        sceneParams['atlas_width'], sceneParams['atlas_height'],
                        sceneParams['atlas_depth'])
                },
                };
                rayMarchUniforms = extend(rayMarchUniforms, sparseGridUniforms);
                fragmentShaderSource =
                    '#define USE_SPARSE_GRID\n' + fragmentShaderSource;
            }
            if (useLargerStepsWhenOccluded) {
                fragmentShaderSource =
                    '#define LARGER_STEPS_WHEN_OCCLUDED\n' + fragmentShaderSource;
            }

            // Bundle uniforms, vertex and fragment shader in a material
            let rayMarchMaterial = new THREE.ShaderMaterial({
                uniforms: rayMarchUniforms,
                vertexShader: rayMarchVertexShader,
                fragmentShader: fragmentShaderSource,
                vertexColors: true,
            });
            rayMarchMaterial.side = THREE.DoubleSide;
            rayMarchMaterial.depthTest = false;
            rayMarchMaterial.needsUpdate = true;

            // Create a proxy plane.
            let fullScreenPlane = new THREE.PlaneGeometry(width, height);
            let fullScreenPlaneMesh =
                new THREE.Mesh(fullScreenPlane, rayMarchMaterial);
            fullScreenPlaneMesh.position.z = -100;
            fullScreenPlaneMesh.frustumCulled = false;

            let gRayMarchScene = new THREE.Scene();
            gRayMarchScene.add(fullScreenPlaneMesh);
            gRayMarchScene.matrixWorldAutoUpdate = false;

            gOrbitControls.update();
            // Render once, then when the camera position (via controls)
            // changes or the window is resized
            renderProgressively(
                gCamera, gRayMarchScene, gRenderer
            )
            gOrbitControls.addEventListener(
                'change', () => renderProgressively(
                    gCamera, gRayMarchScene, gRenderer
                )
            );
        }

        /**
         * Renders the scene
         */
        function renderProgressively(gCamera,
            gRayMarchScene, gRenderer) { 
                
            gCamera.updateProjectionMatrix();
            gCamera.updateMatrixWorld();

            gRenderer.setRenderTarget(gLowResTexture);
            gRenderer.clear();
        
            //
            // For progressive upsampling, jitter the camera matrix within 
            // the pixel footprint.
            //
        
            // We start by forming a set of jitter offsets that touch every 
            // high resolution pixel center.
            const downSamplingFactor =
                gAccumulationTextures[0].width / gLowResTexture.width;
            const isEven = (downSamplingFactor % 2) == 0;
            // These values assume an even downsampling factor.
            let jitterOffset = 0.5;
            let endIndex = Math.trunc(downSamplingFactor / 2);
            if (!isEven) {
                // But it's not that hard to correct for this assumption.
                jitterOffset = 0.5;
                endIndex += 1;
            }
            let samples_x = [];
            let samples_y = [];
            for (let i = 0; i < endIndex; i++) {
                for (let j = 0; j < endIndex; j++) {
                    samples_x.push((jitterOffset + i) / downSamplingFactor);
                    samples_y.push((jitterOffset + j) / downSamplingFactor);
            
                    samples_x.push(-(jitterOffset + i) / downSamplingFactor);
                    samples_y.push((jitterOffset + j) / downSamplingFactor);
            
                    samples_x.push((jitterOffset + i) / downSamplingFactor);
                    samples_y.push(-(jitterOffset + j) / downSamplingFactor);
            
                    samples_x.push(-(jitterOffset + i) / downSamplingFactor);
                    samples_y.push(-(jitterOffset + j) / downSamplingFactor);
                }
            }
        
            // To set up the jitter properly we need to update the projection 
            // matrices of both our cameras in tandem:
            // 1) the orthographic blit matrix that kicks off the ray march,
            // and
            // 2) the perspective projection matrix which computes ray 
            // origins/directions.
            let sample_index = gFrameIndex % samples_x.length;
            let offset_x = samples_x[sample_index];
            let offset_y = samples_y[sample_index];
        
            // First update the orthographic camera, which uses coordinates in
            //   resolution * [-0.5,0,5]^2.
            gLowResBlitCamera.left = offset_x + gLowResTexture.width / -2;
            gLowResBlitCamera.right = offset_x + gLowResTexture.width / 2;
            gLowResBlitCamera.top = offset_y + gLowResTexture.height / 2;
            gLowResBlitCamera.bottom = offset_y + gLowResTexture.height / -2;
            gLowResBlitCamera.updateProjectionMatrix();
        
            // After this we will be working with clip space cameras, that have
            // coordinates in
            //   [-1,1]^2.
            // So we need to scale the offset accordingly.
            offset_x *= 2.0 / gLowResTexture.width;
            offset_y *= 2.0 / gLowResTexture.height;
        
            // Now adjust the projection matrix that computes the ray parameters.
            let clip_T_camera = gCamera.projectionMatrix.clone();
            clip_T_camera.elements[8] += offset_x;
            clip_T_camera.elements[9] += offset_y;
        
            //
            // Now we can do the volume rendering at a lower resolution.
            //
        
            let camera_T_clip = new THREE.Matrix4();
            camera_T_clip = camera_T_clip.copy(clip_T_camera).invert();
        
            let world_T_camera = gCamera.matrixWorld;
            let world_T_clip = new THREE.Matrix4();
            world_T_clip.multiplyMatrices(world_T_camera, camera_T_clip);
        
            gRayMarchScene.children[0].material.uniforms['world_T_clip']
                ['value'] = world_T_clip;
            gRayMarchScene.children[0].material.uniforms['displayMode']
                ['value'] = gDisplayMode - 0;
            gRayMarchScene.children[0].material.uniforms['stepMult']
                ['value'] = gStepMult;
            gRenderer.render(gRayMarchScene, gLowResBlitCamera);
        
            //
            // Finally collect these low resolution samples into our high
            // resolution accumulation bufer.
            //
        
            // With more subsampling we need to average more aggressively over
            // time. This is controled by emaAlpha (exponential moving
            // average), which averages more when the value gets smaller.
            // This formula for setting emaAlpha was hand-tuned to work well
            // in gardenvase.
            let emaAlpha = Math.min(1.0, Math.sqrt(0.1 / samples_x.length));
        
            let accumulationTargetIndex = gFrameIndex % 2;
            let accumulationReadIndex = 1 - accumulationTargetIndex;
            gRenderer.setRenderTarget(
                gAccumulationTextures[accumulationTargetIndex]
            );
            gSceneAccumulate.children[0].material.uniforms['mapHistory']
                ['value'] = 
                gAccumulationTextures[accumulationReadIndex].texture;
            gSceneAccumulate.children[0].material.uniforms['jitterOffset']
                ['value'] = new THREE.Vector2(offset_x, offset_y);
            gSceneAccumulate.children[0].material.uniforms['emaAlpha']
                ['value'] = emaAlpha;
            gRenderer.clear();
            gRenderer.render(gSceneAccumulate, gHighResBlitCamera);
        
            gRenderer.setRenderTarget(null);
            gSceneNormalize.children[0].material.uniforms['map']['value']
                = gAccumulationTextures[accumulationTargetIndex].texture;
            gRenderer.clear();
            gRenderer.render(gSceneNormalize, gHighResBlitCamera);
            
            if (props.showStats) {
                gStats.update();
            }
        }
        
        let lowResFactor = 1;
        let useLargerStepsWhenOccluded = false;

        const nearPlane = 0.2;
        const vfovy = 35;

        const viewspace = viewspaceRef.current;
        if (props.showStats) {
            gStats = Stats();
            viewspace.appendChild(gStats.dom);
            gStats.dom.style.position = 'absolute';
        }
        
        const canvas = canvasRef.current;

        // Set up a high performance WebGL context, making sure that
        // anti-aliasing is turned off.
        let gl = canvas.getContext('webgl2');
        let gRenderer = new THREE.WebGLRenderer({
            alpha: false,
            antialias: false,
            autoClear: false,
            canvas: canvas,
            context: gl,
            depth: false,
            desynchronized: true,
            powerPreference: 'high-performance',
            precision: 'mediump',
            stencil: false,
        });
        let gCamera = new THREE.PerspectiveCamera(
            72,
            Math.trunc(viewspace.offsetWidth / lowResFactor) /
                Math.trunc(viewspace.offsetHeight / lowResFactor),
            nearPlane, 100.0);
        if (document.fullscreenElement) {
            gRenderer.setSize(screen.width, screen.height);
        } else {
            gRenderer.setSize(
                viewspace.offsetWidth, viewspace.offsetHeight
            );
        }
        gRenderer.setPixelRatio(window.devicePixelRatio);
        let rendererWidth = gRenderer.domElement.width;
        let rendererHeight = gRenderer.domElement.height;
        gCamera.aspect = rendererWidth / rendererHeight;

        // Set up the normal scene used for rendering.
        gCamera.position.set(...props.defaultPose.position);
        gCamera.fov = vfovy;
        gHighResBlitCamera = new THREE.OrthographicCamera(
            rendererWidth / -2, rendererWidth / 2,
            rendererHeight / 2, rendererHeight / -2,
            -10000, 10000);
        gHighResBlitCamera.position.z = 100;
        let fullScreenPlane = new THREE.PlaneGeometry(
            rendererWidth, rendererHeight
        );
      
        gLowResTexture = new THREE.WebGLRenderTarget(
            Math.trunc(rendererWidth / lowResFactor),
            Math.trunc(rendererHeight / lowResFactor), {
              minFilter: THREE.NearestFilter,
              magFilter: THREE.NearestFilter,
              type: THREE.UnsignedByteType,
              format: THREE.RGBAFormat
            });
        gAccumulationTextures[0] = new THREE.WebGLRenderTarget(
            rendererWidth, rendererHeight,
            {
                minFilter: THREE.NearestFilter,
                magFilter: THREE.NearestFilter,
                type: THREE.FloatType,
                format: THREE.RGBAFormat
            }
        );
        gAccumulationTextures[1] = new THREE.WebGLRenderTarget(
            rendererWidth, rendererHeight,
            {
                minFilter: THREE.NearestFilter,
                magFilter: THREE.NearestFilter,
                type: THREE.FloatType,
                format: THREE.RGBAFormat
            }
        );
      
        let fullScreenAccumulateQuad = new THREE.Mesh(
            fullScreenPlane,
            createAccumulateMaterial(
                gLowResTexture.texture, gAccumulationTextures[1],
                new THREE.Vector2(
                    Math.trunc(rendererWidth / lowResFactor),
                    Math.trunc(rendererHeight / lowResFactor)),
                new THREE.Vector2(
                    rendererWidth, rendererHeight)
            )
        );
        fullScreenAccumulateQuad.position.z = -100;
        gSceneAccumulate = new THREE.Scene();
        gSceneAccumulate.add(fullScreenAccumulateQuad);
        gSceneAccumulate.matrixWorldAutoUpdate = false;
      
        let fullScreenNormalizeQuad = new THREE.Mesh(
            fullScreenPlane,
            createNormalizeMaterial(gAccumulationTextures[0].texture));
        fullScreenNormalizeQuad.position.z = -100;
        gSceneNormalize = new THREE.Scene();
        gSceneNormalize.add(fullScreenNormalizeQuad);
        gSceneNormalize.matrixWorldAutoUpdate = false;
      
        gLowResBlitCamera = new THREE.OrthographicCamera(
            Math.trunc(rendererWidth / lowResFactor) / -2,
            Math.trunc(rendererWidth / lowResFactor) / 2,
            Math.trunc(rendererHeight / lowResFactor) / 2,
            Math.trunc(rendererHeight / lowResFactor) / -2,
            -10000, 10000);
        gLowResBlitCamera.position.z = 100;

        gOrbitControls = new OrbitControls(gCamera, canvas);
        // Disable damping until we have temporal reprojection for upscaling.
        // gOrbitControls.enableDamping = true;
        gOrbitControls.screenSpacePanning = true;
        gOrbitControls.zoomSpeed = 0.5;
        gOrbitControls.target.x = props.defaultPose.lookat[0];
        gOrbitControls.target.y = props.defaultPose.lookat[1];
        gOrbitControls.target.z = props.defaultPose.lookat[2];
        loadScene(
            Math.trunc(rendererWidth / lowResFactor),
            Math.trunc(rendererHeight / lowResFactor),
            useLargerStepsWhenOccluded,
            nearPlane, gCamera, gOrbitControls,
            gRenderer);

        function onResize() {
            const viewspace = viewspaceRef.current;
            gRenderer.setSize(
                viewspace.clientWidth, viewspace.clientHeight
            );
            gCamera.aspect = viewspace.clientWidth /
                viewspace.clientHeight;
            gCamera.updateProjectionMatrix();
        };
        window.addEventListener('resize', onResize);
        function onFullscreenChange() {
            if (document.fullscreenElement) {
                setCurrentlyFullScreen(true);
                gCamera.aspect = window.screen.width /
                    window.screen.height;
                gCamera.updateProjectionMatrix();
                gRenderer.setSize(
                    window.screen.width, window.screen.height
                );
            } else {
                setCurrentlyFullScreen(false);
                const viewspaceContainer = viewspaceContainerRef.current;
                gRenderer.setSize(
                    viewspaceContainer.clientWidth, viewspaceContainer.clientHeight
                );
                gCamera.aspect = viewspaceContainer.clientWidth /
                    viewspaceContainer.clientHeight;
                gCamera.updateProjectionMatrix();
            }
        };
        window.addEventListener('fullscreenchange', onFullscreenChange);

        // Cleanup the event listeners when the component is unmounted.
        return () => {
            window.removeEventListener('resize', onResize);
            document.removeEventListener('fullscreenchange', onFullscreenChange);
        };

    }, [props, currentlyFullScreen])

    return (
        <Box
            ref={viewspaceContainerRef}
            sx={{
                height: '100%',
                width: '100%'
            }}
        >
            <Box
                ref={viewspaceRef}
                sx={{
                    display: 'inline-block',
                    height: '100%',
                    position: 'relative',
                    width: '100%'
                }}
            >
                <canvas
                    ref={canvasRef}
                    style={{
                        display: 'block',
                        height: '100%',
                        width: '100%'
                    }}
                />
                <Box>
                    <Box
                        sx={{
                            bottom: 0,
                            position: 'absolute',
                            right: 0
                        }}
                    >
                        <IconButton
                            onClick={toggleFullscreen}
                        >
                        {currentlyFullScreen ?
                            <FullscreenExitIcon />
                        :
                            <FullscreenIcon />
                        }
                        </IconButton>
                    </Box>
                </Box>
            </Box>
        </Box>
    )
}