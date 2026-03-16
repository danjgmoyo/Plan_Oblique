# ---------------------------------------------------------------------------
# PlanOblique_QGIS.py  –  QGIS Processing Script
#
# PLANOBLIQUIFIER by Bernhard Jenny (RMIT University)
# Applies shearing to a terrain model along a chosen azimuth direction.
# Original Python: Bojan Savric (Esri), 10.6.2016
#
# Extended for cartographic and geology work.
# Ported and extended by Claude (Anthropic)
#
# References:
#   Jenny, B. & Patterson, T. (2007). Introducing Plan Oblique Relief.
#       Cartographic Perspectives, 57, 21-40.
#   Jenny, B. et al. (2015). Plan oblique relief for web maps.
#       Cartography and Geographic Information Science, 42(5), 410-418.
#
# HOW TO USE
# ----------
# Processing Toolbox → Scripts → Add Script to Toolbox...
# Browse to this file → double-click "Plan Oblique Relief" to open dialog.
#
# DEPENDENCIES  (all bundled with QGIS / OSGeo4W)
#   gdal, numpy, scipy
# ---------------------------------------------------------------------------

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingOutputRasterLayer,
)

import math
import numpy as np
from scipy.ndimage import gaussian_filter
from osgeo import gdal

gdal.UseExceptions()


class PlanObliqueAlgorithm(QgsProcessingAlgorithm):
    """
    Plan oblique relief after Bernhard Jenny (RMIT University).

    PARAMETERS
    ----------
    Inclination angle (1-89°)
        Controls the viewing angle. 45° = classic oblique. Lower values
        produce more dramatic side views. Higher values approach top-down.

    Shear azimuth (0-360°)
        Direction the terrain shears toward, in degrees clockwise from north.
          0°  = shear north  (default, classic look, terrain rises upward)
         180° = shear south  (terrain rises downward — unusual but valid)
          90° = shear east   (good for N-S mountain chains)
         270° = shear west
        Jenny (2007) notes the shear direction is just as important as the
        angle for visual clarity of different terrain orientations.

    Vertical exaggeration (default 1.0)
        Scales elevation values before shearing. Values > 1 stretch vertical
        relief, making subtle structures more visible. Essential for geology
        work on low-relief terrain. Independent of inclination angle.

    Pre-smoothing sigma (default 0 = off)
        Gaussian sigma applied to the DEM before shearing. Jenny recommends
        light smoothing (sigma 0.5-1.5) for small-scale cartographic use to
        reduce pixel noise in the output. Leave at 0 for raw DEM output.

    Clip to input extent (default False)
        When True, the output is cropped back to the original DEM extent.
        Useful for geology work where the output must overlay exactly with
        other data layers on the same grid. When False (default), the output
        is taller than the input: high terrain shifts northward into extra
        rows added above the original extent.

    OUTPUT PRODUCTS
    ---------------
    Oblique DEM
        The sheared elevation raster. Load this into QGIS and apply any
        hillshade, hypsometric tint, or colour ramp you prefer. The raster
        stores elevation values displaced to their sheared positions, so a
        standard QGIS hillshade run on this output will correctly shade the
        oblique terrain.

    Occlusion mask (optional)
        A binary raster (1 = visible, 0 = occluded) marking pixels that are
        hidden behind taller sheared terrain. Useful in geology for identifying
        which parts of a structure are not visible from the oblique viewpoint,
        and in cartography to decide where to place labels or symbols.
    """

    INPUT       = "INPUT"
    OUTPUT      = "OUTPUT"
    ANGLE       = "ANGLE"
    AZIMUTH     = "AZIMUTH"
    V_EXAG      = "V_EXAG"
    SMOOTH      = "SMOOTH"
    CLIP_EXTENT = "CLIP_EXTENT"
    OUTPUT_OCC  = "OUTPUT_OCC"

    def tr(self, s):
        return QCoreApplication.translate("PlanOblique", s)

    def createInstance(self):
        return PlanObliqueAlgorithm()

    def name(self):
        return "planobliquerelief"

    def displayName(self):
        return self.tr("Plan Oblique Relief")

    def group(self):
        return self.tr("Terrain Tools")

    def groupId(self):
        return "terraintools"

    def shortHelpString(self):
        return self.tr(
            "Plan oblique relief after Bernhard Jenny (RMIT University).\n\n"
            "Shears a DEM so elevation displaces pixels in the chosen azimuth "
            "direction, creating a side-on terrain view on a planimetric map.\n\n"
            "INCLINATION ANGLE\n"
            "  45° = classic oblique (default)\n"
            "  30° = more dramatic side view\n"
            "  60° = flatter, closer to top-down\n\n"
            "SHEAR AZIMUTH\n"
            "  Direction terrain shears toward (degrees clockwise from north).\n"
            "  0° = north (default). Use 90° or 270° for N-S mountain chains.\n\n"
            "VERTICAL EXAGGERATION\n"
            "  Scales elevation before shearing. >1.0 exaggerates relief.\n"
            "  Useful for low-relief terrain in geology work.\n\n"
            "PRE-SMOOTHING SIGMA\n"
            "  Gaussian smooth applied before shearing (0 = off).\n"
            "  Jenny recommends sigma 0.5-1.5 for small-scale cartographic use.\n\n"
            "CLIP TO INPUT EXTENT\n"
            "  When enabled, output is cropped to the original DEM footprint.\n"
            "  Use this when the output must overlay other data exactly.\n\n"
            "OCCLUSION MASK (optional)\n"
            "  Binary raster: 1 = visible, 0 = hidden behind taller terrain.\n"
            "  Useful in geology to see which structures are occluded."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, self.tr("Input DEM")))

        self.addParameter(QgsProcessingParameterNumber(
            self.ANGLE, self.tr("Inclination angle (degrees)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=45.0, minValue=1.0, maxValue=89.9))

        self.addParameter(QgsProcessingParameterNumber(
            self.AZIMUTH, self.tr("Shear azimuth (degrees clockwise from north)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0, minValue=0.0, maxValue=360.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.V_EXAG, self.tr("Vertical exaggeration"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0, minValue=0.1, maxValue=20.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.SMOOTH, self.tr("Pre-smoothing sigma (0 = off)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0, minValue=0.0, maxValue=10.0))

        self.addParameter(QgsProcessingParameterBoolean(
            self.CLIP_EXTENT,
            self.tr("Clip output to original DEM extent"),
            defaultValue=False))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, self.tr("Plan oblique DEM")))

        # Occlusion mask is optional — user can leave it blank
        occ_param = QgsProcessingParameterRasterDestination(
            self.OUTPUT_OCC, self.tr("Occlusion mask (optional)"),
            optional=True, createByDefault=False)
        self.addParameter(occ_param)

    # ------------------------------------------------------------------
    # Internal: read raster → float64 array + metadata
    # ------------------------------------------------------------------
    def _read(self, path):
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException("Cannot open: {}".format(path))
        band  = ds.GetRasterBand(1)
        arr   = band.ReadAsArray().astype(np.float64)
        nd    = band.GetNoDataValue()
        gt    = ds.GetGeoTransform()
        proj  = ds.GetProjection()
        ds    = None
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr, gt, proj

    # ------------------------------------------------------------------
    # Internal: write float32 GeoTIFF
    # ------------------------------------------------------------------
    def _write(self, path, arr, gt, proj, nodata=-9999.0):
        rows, cols = arr.shape
        drv    = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(path, cols, rows, 1, gdal.GDT_Float32,
                            ["COMPRESS=LZW", "TILED=YES"])
        if out_ds is None:
            raise QgsProcessingException("Cannot create: {}".format(path))
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        band = out_ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(
            np.where(np.isnan(arr), nodata, arr).astype(np.float32))
        band.FlushCache()
        out_ds.FlushCache()
        out_ds = None

    # ------------------------------------------------------------------
    # Internal: rotate array and geotransform for non-north azimuths
    # ------------------------------------------------------------------
    def _rotate_array(self, arr, azimuth_deg):
        """
        Rotate the DEM so the shear always acts northward in array space,
        then we rotate the result back.
        np.rot90 handles 0/90/180/270 exactly.
        For arbitrary angles we use scipy.ndimage.rotate (nearest+no reshape).
        Only 0/90/180/270 keep the geotransform simple; arbitrary angles
        introduce a rotated bounding box which we handle in _rotate_gt.
        Returns (rotated_array, applied_rotation_degrees).
        """
        # Snap to nearest 90° for the four cardinal directions
        snap = round(azimuth_deg / 90.0) * 90.0 % 360.0
        if abs(azimuth_deg - snap) < 0.5:
            k = int(snap / 90)   # number of 90° CCW rotations
            # azimuth 0 (north) → no rotation (k=0)
            # azimuth 90 (east) → rotate CCW 90° so east becomes up (k=1... wait)
            # We want: the shear azimuth to become "up" in array space (row 0).
            # Array row 0 = north. East = column direction.
            # To shear east: rotate array 90° CW (= 3 CCW) so east→north
            rot_map = {0: 0, 90: 3, 180: 2, 270: 1}
            k_ccw = rot_map.get(int(snap), 0)
            return np.rot90(arr, k=k_ccw), snap
        else:
            from scipy.ndimage import rotate as nd_rotate
            # Positive azimuth CW → rotate array CCW by same amount
            # so the shear direction aligns with north (row 0)
            rotated = nd_rotate(arr, angle=azimuth_deg,
                                reshape=True, order=1, cval=np.nan)
            return rotated, azimuth_deg

    def _unrotate_array(self, arr, applied_rotation):
        """Reverse the rotation applied by _rotate_array."""
        snap = round(applied_rotation / 90.0) * 90.0 % 360.0
        if abs(applied_rotation - snap) < 0.5:
            rot_map_rev = {0: 0, 90: 1, 180: 2, 270: 3}
            k_ccw = rot_map_rev.get(int(snap), 0)
            return np.rot90(arr, k=k_ccw)
        else:
            from scipy.ndimage import rotate as nd_rotate
            return nd_rotate(arr, angle=-applied_rotation,
                             reshape=True, order=1, cval=np.nan)

    # ------------------------------------------------------------------
    # Core shearing algorithm (column-by-column, unchanged from Jenny)
    # ------------------------------------------------------------------
    def _shear(self, arr, cell_size, ref_elev, elev_scale, feedback):
        """
        Returns (arrOut, occOut) where occOut is a boolean array:
        True = pixel is visible, False = occluded by taller terrain.
        """
        nRows_orig, nCols = arr.shape
        north = nRows_orig * cell_size   # working in pixel units

        max_dy = (np.nanmax(arr) - ref_elev) * elev_scale
        dRows  = int(max_dy / cell_size + 1)
        nRows  = nRows_orig + dRows
        north_ext = north + dRows * cell_size

        # Enlarged input: original data at bottom, NaN rows at top
        arr_ext = np.full((nRows, nCols), np.nan, dtype=np.float64)
        arr_ext[dRows:, :] = arr

        arrOut = np.full((nRows, nCols), np.nan, dtype=np.float64)
        occOut = np.zeros((nRows, nCols), dtype=np.uint8)

        report_every = max(1, nCols // 20)

        for col in range(nCols):
            if feedback.isCanceled():
                return None, None

            if col % report_every == 0:
                pct = 20 + int(65 * col / nCols)
                feedback.setProgress(pct)

            # Find first valid row from the bottom
            prevRow = nRows - 1
            prevZ   = np.nan

            for row in range(prevRow, -1, -1):
                if math.isnan(arr_ext[row, col]):
                    arrOut[row, col] = np.nan
                else:
                    prevRow = row
                    prevZ   = arr_ext[row, col]
                    break

            prevShearedY = north_ext - prevRow * cell_size

            for row in range(prevRow, -1, -1):
                targetY       = north_ext - row * cell_size
                interpolatedZ = np.nan
                occluded      = False

                for r in range(prevRow, -1, -1):
                    z = arr_ext[r, col]

                    if math.isnan(z):
                        move = r - 1
                        while move >= 0 and math.isnan(arr_ext[move, col]):
                            move -= 1
                        prevRow       = move
                        interpolatedZ = np.nan
                        prevZ         = np.nan
                        break

                    shearedY = (north_ext - r * cell_size +
                                (z - ref_elev) * elev_scale)

                    if shearedY > targetY:
                        denom = shearedY - prevShearedY
                        if denom != 0.0:
                            w = (targetY - prevShearedY) / denom
                            interpolatedZ = w * z + (1.0 - w) * prevZ
                        else:
                            interpolatedZ = z
                        break

                    prevRow = r

                    if shearedY >= prevShearedY:
                        prevShearedY = shearedY
                        prevZ        = z
                    else:
                        # shearedY < prevShearedY means this vertex is
                        # occluded by a previously seen higher vertex
                        occluded = True

                arrOut[row, col] = interpolatedZ
                if occluded and not math.isnan(interpolatedZ):
                    occOut[row, col] = 0
                elif not math.isnan(interpolatedZ):
                    occOut[row, col] = 1

        return arrOut, occOut, dRows, nRows

    # ------------------------------------------------------------------
    # Crop empty rows and return corrected geotransform
    # ------------------------------------------------------------------
    def _crop_and_gt(self, arrOut, occOut, gt, cell_size,
                     dRows, nRows, clip_to_input, nRows_orig):
        """
        Crop empty rows from top and bottom.
        If clip_to_input=True, further crop to the original row count.
        Returns (cropped_dem, cropped_occ, new_geotransform).
        """
        mask      = np.isnan(arrOut)
        row_data  = (~mask).sum(axis=1)
        valid_rows = np.flatnonzero(row_data)

        if len(valid_rows) == 0:
            raise QgsProcessingException(
                "Shearing produced no valid output pixels.")

        start = int(valid_rows[0])
        end   = int(valid_rows[-1]) + 1

        if clip_to_input:
            # The original rows sit at indices dRows..nRows in the extended
            # array. Clip to that range (intersected with valid data range).
            start = max(start, dRows)
            end   = min(end, dRows + nRows_orig)

        arrOut_c = arrOut[start:end, :]
        occOut_c = occOut[start:end, :].astype(np.float32)
        # Mark nodata pixels in occlusion mask
        occOut_c[np.isnan(arrOut[start:end, :])] = np.nan

        # Correct y-origin: shift top edge down by `start` rows
        x_orig   = gt[0]
        y_orig   = gt[3] + (gt[5] * -dRows)   # extended north edge
        new_y    = y_orig - start * cell_size
        new_gt   = (x_orig, cell_size, 0.0, new_y, 0.0, -cell_size)

        return arrOut_c, occOut_c, new_gt

    # ------------------------------------------------------------------
    # processAlgorithm
    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):

        dem_layer   = self.parameterAsRasterLayer(
            parameters, self.INPUT, context)
        angle       = self.parameterAsDouble(parameters, self.ANGLE, context)
        azimuth     = self.parameterAsDouble(parameters, self.AZIMUTH, context) % 360.0
        v_exag      = self.parameterAsDouble(parameters, self.V_EXAG, context)
        smooth_sig  = self.parameterAsDouble(parameters, self.SMOOTH, context)
        clip_extent = self.parameterAsBool(parameters, self.CLIP_EXTENT, context)
        out_path    = self.parameterAsOutputLayer(
            parameters, self.OUTPUT, context)
        out_occ     = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_OCC, context)
        make_occ    = bool(out_occ)

        dem_path = dem_layer.source()

        feedback.pushInfo("Plan Oblique Relief")
        feedback.pushInfo("  Input      : {}".format(dem_path))
        feedback.pushInfo("  Angle      : {}°".format(angle))
        feedback.pushInfo("  Azimuth    : {}°".format(azimuth))
        feedback.pushInfo("  V. exag    : {}x".format(v_exag))
        feedback.pushInfo("  Smoothing  : sigma={}".format(smooth_sig))
        feedback.pushInfo("  Clip extent: {}".format(clip_extent))
        feedback.pushInfo("  Occ. mask  : {}".format(make_occ))

        # --- 1. Read DEM -------------------------------------------
        feedback.setProgress(5)
        feedback.pushInfo("  [1/6] Reading DEM...")
        arr, gt, proj = self._read(dem_path)
        nRows_orig, nCols_orig = arr.shape
        cell_size = abs(gt[1])

        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            raise QgsProcessingException("DEM has no valid data.")
        feedback.pushInfo("  Grid: {}×{}  Cell: {:.4f}  Elev: {:.1f}–{:.1f}".format(
            nRows_orig, nCols_orig, cell_size, valid.min(), valid.max()))

        # --- 2. Pre-smoothing (optional) ---------------------------
        if smooth_sig > 0:
            feedback.setProgress(8)
            feedback.pushInfo("  [2/6] Pre-smoothing (sigma={})...".format(
                smooth_sig))
            nan_mask = np.isnan(arr)
            arr_filled = np.where(nan_mask, 0.0, arr)
            # Smooth both data and a weight mask, then divide to avoid
            # blurring across nodata boundaries
            weight = (~nan_mask).astype(np.float64)
            arr_sm = gaussian_filter(arr_filled, sigma=smooth_sig)
            w_sm   = gaussian_filter(weight,     sigma=smooth_sig)
            arr    = np.where(w_sm > 0.01, arr_sm / w_sm, np.nan)
            arr[nan_mask] = np.nan
        else:
            feedback.pushInfo("  [2/6] Pre-smoothing skipped.")

        # --- 3. Apply vertical exaggeration ------------------------
        if v_exag != 1.0:
            feedback.setProgress(10)
            feedback.pushInfo("  [3/6] Applying vertical exaggeration ({})...".format(
                v_exag))
            # Scale elevations around the minimum so the base stays fixed
            ref = float(np.nanmin(arr))
            arr = ref + (arr - ref) * v_exag
        else:
            feedback.pushInfo("  [3/6] Vertical exaggeration = 1.0 (skipped).")

        # --- 4. Rotate for azimuth ---------------------------------
        feedback.setProgress(12)
        feedback.pushInfo("  [4/6] Rotating for azimuth {}°...".format(azimuth))

        if abs(azimuth) < 0.5 or abs(azimuth - 360.0) < 0.5:
            # Pure north shear — no rotation needed
            arr_rot    = arr
            applied_az = 0.0
        else:
            arr_rot, applied_az = self._rotate_array(arr, azimuth)

        nRows_rot, nCols_rot = arr_rot.shape

        # Recalculate valid range after exaggeration + rotation
        valid_rot = arr_rot[~np.isnan(arr_rot)]
        if len(valid_rot) == 0:
            raise QgsProcessingException("No valid data after rotation.")
        ref_elev  = float(valid_rot.min())
        elev_scale = 1.0 / math.tan(math.radians(angle))

        # --- 5. Shear ---------------------------------------------- 
        feedback.setProgress(15)
        feedback.pushInfo("  [5/6] Shearing ({} cols, scale={:.4f})...".format(
            nCols_rot, elev_scale))

        result = self._shear(arr_rot, cell_size, ref_elev,
                             elev_scale, feedback)
        if result[0] is None:
            return {}   # cancelled

        arrOut, occOut, dRows, nRows_ext = result

        if feedback.isCanceled():
            return {}

        # Un-rotate if we rotated
        if abs(applied_az) > 0.5:
            arrOut = self._unrotate_array(arrOut, applied_az)
            occOut_f = self._unrotate_array(
                occOut.astype(np.float64), applied_az)
            occOut = (occOut_f > 0.5).astype(np.uint8)

        # --- 6. Crop and write -------------------------------------
        feedback.setProgress(88)
        feedback.pushInfo("  [6/6] Cropping and writing outputs...")

        arrOut_c, occOut_c, new_gt = self._crop_and_gt(
            arrOut, occOut, gt, cell_size,
            dRows, nRows_ext, clip_extent, nRows_rot)

        self._write(out_path, arrOut_c, new_gt, proj)
        feedback.pushInfo("  Oblique DEM → {}".format(out_path))

        results = {self.OUTPUT: out_path}

        if make_occ:
            self._write(out_occ, occOut_c, new_gt, proj, nodata=-9999.0)
            feedback.pushInfo("  Occlusion mask → {}".format(out_occ))
            results[self.OUTPUT_OCC] = out_occ

        feedback.setProgress(100)
        feedback.pushInfo("Done.")
        return results