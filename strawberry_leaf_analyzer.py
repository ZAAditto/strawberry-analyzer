"""
Strawberry Leaf Health Analyzer - STREAMLINED VERSION
======================================================
Focused on realistic phone camera detection of important strawberry issues
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from sklearn.cluster import KMeans
import json
import base64
import sys


class StrawberryLeafAnalyzer:
    """Analyzes strawberry leaf images for key health indicators."""
    
    def __init__(self):
        self.color_ranges = {
            'healthy_green': {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},
            'yellow_chlorosis': {'lower': np.array([20, 40, 40]), 'upper': np.array([35, 255, 255])},
            'brown_necrosis': {'lower': np.array([10, 50, 20]), 'upper': np.array([20, 255, 200])},
            'white_mildew': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},
            'dark_spots': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 60])},
        }

    def load_image(self, image_path=None, base64_string=None):
        """Load image from file path or base64 string"""
        if base64_string:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_path:
            image = cv2.imread(image_path)
        else:
            raise ValueError("Must provide either image_path or base64_string")
        
        if image is None:
            raise ValueError("Could not load image")
        return image

    def preprocess_image(self, image):
        """Preprocess image for analysis"""
        max_dim = 1000
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def segment_leaf(self, image):
        """Segment the leaf from the background"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for different leaf colors
        green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 20, 20]), np.array([35, 255, 255]))
        brown_mask = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([25, 255, 200]))
        
        leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)
        leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(leaf_mask)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            return final_mask, largest_contour
        
        return leaf_mask, None

    def analyze_color_indicators(self, image, mask):
        """Analyze color-based health indicators"""
        results = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        leaf_pixels_hsv = hsv[mask > 0]
        leaf_pixels_rgb = rgb[mask > 0]
        leaf_pixels_lab = lab[mask > 0]
        
        if len(leaf_pixels_rgb) == 0:
            return self._get_default_color_results()
        
        # Normalized RGB
        rgb_sum = leaf_pixels_rgb.astype(float).sum(axis=1, keepdims=True)
        rgb_sum[rgb_sum == 0] = 1
        normalized_rgb = leaf_pixels_rgb.astype(float) / rgb_sum
        r_norm, g_norm, b_norm = normalized_rgb[:, 0], normalized_rgb[:, 1], normalized_rgb[:, 2]
        
        # 1. Chlorophyll Content (Overall Green)
        exg = 2 * g_norm - r_norm - b_norm
        exg_mean = np.mean(exg)
        exg_score = np.clip((exg_mean - 0.05) / 0.3 * 100, 0, 100)
        
        results['chlorophyll'] = {
            'value': 'Healthy' if exg_score > 60 else ('Moderate' if exg_score > 35 else 'Low (Nitrogen Deficiency)'),
            'score': float(exg_score),
            'confidence': min(95, 75 + exg_score * 0.2),
            'isHealthy': exg_score > 50,
            'metrics': {'exg_index': float(exg_mean), 'green_dominance': float(np.mean(g_norm))},
            'explanation': f"Overall green color indicates chlorophyll and nitrogen status.",
            'evidence': [
                f"Excess Green Index: {exg_mean:.3f}",
                f"Green dominance: {np.mean(g_norm)*100:.1f}%"
            ],
            'recommendation': 'Chlorophyll levels healthy.' if exg_score > 50 else 
                            'Apply nitrogen fertilizer (urea or ammonium nitrate).'
        }
        
        # 2. Yellow Discoloration (Chlorosis)
        yellow_mask = cv2.inRange(hsv, self.color_ranges['yellow_chlorosis']['lower'], 
                                  self.color_ranges['yellow_chlorosis']['upper'])
        yellow_mask = cv2.bitwise_and(yellow_mask, mask)
        yellow_ratio = np.sum(yellow_mask > 0) / np.sum(mask > 0) * 100
        
        results['chlorosis'] = {
            'value': 'None' if yellow_ratio < 10 else (f'Mild ({yellow_ratio:.1f}%)' if yellow_ratio < 25 else f'Severe ({yellow_ratio:.1f}%)'),
            'score': float(max(0, 100 - yellow_ratio * 3)),
            'confidence': min(96, 80 + (100 - yellow_ratio) * 0.16),
            'isHealthy': yellow_ratio < 15,
            'metrics': {'yellow_percentage': float(yellow_ratio)},
            'explanation': "Yellowing indicates nutrient deficiency (N, Fe) or disease.",
            'evidence': [f"Yellow coverage: {yellow_ratio:.1f}%"],
            'recommendation': 'No significant yellowing.' if yellow_ratio < 15 else
                            'Check nitrogen and iron levels. Apply chelated iron if needed.'
        }
        
        # 3. Brown Patches (Necrosis)
        brown_mask = cv2.inRange(hsv, self.color_ranges['brown_necrosis']['lower'],
                                 self.color_ranges['brown_necrosis']['upper'])
        brown_mask = cv2.bitwise_and(brown_mask, mask)
        necrosis_ratio = np.sum(brown_mask > 0) / np.sum(mask > 0) * 100
        
        results['necrosis'] = {
            'value': 'None' if necrosis_ratio < 5 else f'Present ({necrosis_ratio:.1f}% coverage)',
            'score': float(max(0, 100 - necrosis_ratio * 5)),
            'confidence': min(97, 82 + (100 - necrosis_ratio) * 0.15),
            'isHealthy': necrosis_ratio < 8,
            'metrics': {'brown_percentage': float(necrosis_ratio)},
            'explanation': "Dead brown tissue from disease, potassium deficiency, or salt damage.",
            'evidence': [f"Necrotic tissue: {necrosis_ratio:.1f}%"],
            'recommendation': 'No necrosis detected.' if necrosis_ratio < 8 else
                            'Remove affected leaves. Check potassium levels and avoid salt buildup.'
        }
        
        # 4. Interveinal Chlorosis (Iron/Magnesium deficiency)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        edges = cv2.Canny(gray_masked, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        vein_area = cv2.bitwise_and(mask, dilated_edges)
        non_vein_area = cv2.bitwise_and(mask, cv2.bitwise_not(dilated_edges))
        
        if np.sum(vein_area > 0) > 0 and np.sum(non_vein_area > 0) > 0:
            vein_green = np.mean(hsv[vein_area > 0, 0])
            non_vein_green = np.mean(hsv[non_vein_area > 0, 0])
            interveinal_diff = abs(vein_green - non_vein_green)
        else:
            interveinal_diff = 0
        
        results['interveinal'] = {
            'value': 'Not Present' if interveinal_diff <= 15 else 'Detected (Iron/Mg Deficiency)',
            'score': float(max(0, 100 - interveinal_diff * 3)),
            'confidence': min(88, 70 + (100 - interveinal_diff) * 0.18),
            'isHealthy': interveinal_diff <= 15,
            'metrics': {'vein_interveinal_difference': float(interveinal_diff)},
            'explanation': "Yellowing between veins while veins stay green indicates Fe or Mg deficiency.",
            'evidence': [f"Vein/tissue color difference: {interveinal_diff:.1f}°"],
            'recommendation': 'No interveinal chlorosis.' if interveinal_diff <= 15 else
                            'Apply chelated iron (Fe-EDDHA) or Epsom salt (magnesium sulfate).'
        }
        
        # 5. Marginal Necrosis (Potassium deficiency)
        edge_kernel = np.ones((15, 15), np.uint8)
        eroded_mask = cv2.erode(mask, edge_kernel, iterations=1)
        margin_mask = cv2.subtract(mask, eroded_mask)
        margin_brown = cv2.bitwise_and(brown_mask, margin_mask)
        marginal_ratio = np.sum(margin_brown > 0) / max(1, np.sum(margin_mask > 0)) * 100
        
        results['marginal'] = {
            'value': 'None' if marginal_ratio < 10 else f'Edge Browning ({marginal_ratio:.1f}%)',
            'score': float(max(0, 100 - marginal_ratio * 4)),
            'confidence': min(95, 78 + (100 - marginal_ratio) * 0.17),
            'isHealthy': marginal_ratio < 15,
            'metrics': {'edge_browning_percentage': float(marginal_ratio)},
            'explanation': "Brown edges indicate potassium deficiency or salt burn.",
            'evidence': [f"Edge browning: {marginal_ratio:.1f}%"],
            'recommendation': 'Leaf margins healthy.' if marginal_ratio < 15 else
                            'Apply potassium sulfate fertilizer. Check soil EC to rule out salt damage.'
        }
        
        return results

    def _get_default_color_results(self):
        """Return default results when analysis fails"""
        default = {
            'value': 'Unable to analyze', 'score': 0, 'confidence': 0, 'isHealthy': False,
            'metrics': {}, 'explanation': 'Insufficient leaf pixels detected.',
            'evidence': ['Leaf segmentation failed'], 'recommendation': 'Upload a clearer image.'
        }
        return {key: default.copy() for key in ['chlorophyll', 'chlorosis', 'necrosis', 'interveinal', 'marginal']}

    def analyze_structural_features(self, image, mask, contour):
        """Analyze structural features"""
        results = {}
        
        if contour is None:
            return self._get_default_structural_results()
        
        leaf_area = cv2.contourArea(contour)
        leaf_perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        image_area = image.shape[0] * image.shape[1]
        relative_size = (leaf_area / image_area) * 100
        size_normal = 15 < relative_size < 70
        
        # 1. Leaf Size
        results['leafSize'] = {
            'value': 'Normal' if size_normal else ('Large' if relative_size > 70 else 'Small'),
            'score': float(100 if size_normal else max(0, 100 - abs(relative_size - 40) * 2)),
            'confidence': 75,  # Lower confidence without reference object
            'isHealthy': size_normal,
            'metrics': {'relative_size_percent': float(relative_size)},
            'explanation': "Leaf size relative to image frame (note: requires consistent photo distance).",
            'evidence': [f"Leaf occupies {relative_size:.1f}% of image"],
            'recommendation': 'Leaf size appears normal.' if size_normal else 'Check overall plant vigor and root health.'
        }
        
        # 2. Shape Distortion
        circularity = 4 * np.pi * leaf_area / (leaf_perimeter ** 2) if leaf_perimeter > 0 else 0
        solidity = leaf_area / hull_area if hull_area > 0 else 0
        shape_normal = 0.3 < circularity < 0.8 and solidity > 0.75
        
        results['shapeDistortion'] = {
            'value': 'Normal' if shape_normal else 'Distorted (Check for Viruses/Mites)',
            'score': float(min(100, (circularity * 50 + solidity * 50) * 1.2)),
            'confidence': min(90, 75 + solidity * 15),
            'isHealthy': shape_normal,
            'metrics': {'solidity': float(solidity)},
            'explanation': "Abnormal shape may indicate viral infection or pest damage.",
            'evidence': [f"Shape integrity: {solidity:.2f}"],
            'recommendation': 'Leaf shape normal.' if shape_normal else 
                            'Inspect for two-spotted spider mites or viral symptoms. Remove infected plants.'
        }
        
        # 3. Edge Irregularities
        hull_perimeter = cv2.arcLength(hull, True)
        edge_irreg = abs(leaf_perimeter - hull_perimeter) / hull_perimeter if hull_perimeter > 0 else 0
        edge_normal = 0.1 < edge_irreg < 0.5
        
        results['edgeIrreg'] = {
            'value': 'Normal Serration' if edge_normal else ('Smooth/Damaged' if edge_irreg < 0.1 else 'Highly Irregular'),
            'score': float(max(0, 100 - abs(edge_irreg - 0.3) * 200)),
            'confidence': min(88, 70 + (100 - abs(edge_irreg - 0.3) * 100) * 0.18),
            'isHealthy': edge_normal,
            'metrics': {'edge_irregularity': float(edge_irreg)},
            'explanation': "Edge pattern indicates pest damage or disease.",
            'evidence': [f"Edge irregularity: {edge_irreg:.2f}"],
            'recommendation': 'Edge pattern normal.' if edge_normal else 'Check for caterpillar or beetle damage.'
        }
        
        return results

    def _get_default_structural_results(self):
        default = {'value': 'Unable to analyze', 'score': 0, 'confidence': 0, 'isHealthy': False,
                   'metrics': {}, 'explanation': 'Contour not detected.', 'evidence': [], 'recommendation': 'Upload clearer image.'}
        return {key: default.copy() for key in ['leafSize', 'shapeDistortion', 'edgeIrreg']}

    def analyze_disease_symptoms(self, image, mask):
        """Analyze major disease symptoms"""
        results = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Leaf Spot Detection (Most Common)
        dark_mask = cv2.inRange(hsv, self.color_ranges['dark_spots']['lower'], self.color_ranges['dark_spots']['upper'])
        brown_mask = cv2.inRange(hsv, self.color_ranges['brown_necrosis']['lower'], self.color_ranges['brown_necrosis']['upper'])
        spot_mask = cv2.bitwise_and(cv2.bitwise_or(dark_mask, brown_mask), mask)
        
        contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if 20 < area < 5000 and perim > 0:
                circularity = 4 * np.pi * area / (perim ** 2)
                if circularity > 0.3:
                    spot_count += 1
        
        results['leafSpot'] = {
            'value': 'Not Detected' if spot_count < 3 else f'{spot_count} Spot(s) Detected',
            'score': float(max(0, 100 - spot_count * 10)),
            'confidence': min(95, 75 + (10 - min(spot_count, 10)) * 2),
            'isHealthy': spot_count < 5,
            'metrics': {'spot_count': spot_count},
            'explanation': "Circular spots indicate fungal disease (Mycosphaerella, Phomopsis, etc.).",
            'evidence': [f"Detected spots: {spot_count}"],
            'recommendation': 'No significant leaf spots.' if spot_count < 5 else 
                            'Remove affected leaves. Apply captan or copper fungicide. Improve air circulation.'
        }
        
        # 2. Powdery Mildew (Very Common)
        white_mask = cv2.inRange(hsv, self.color_ranges['white_mildew']['lower'], self.color_ranges['white_mildew']['upper'])
        white_mask = cv2.bitwise_and(white_mask, mask)
        mildew_ratio = np.sum(white_mask > 0) / max(1, np.sum(mask > 0)) * 100
        
        results['powderyMildew'] = {
            'value': 'Not Detected' if mildew_ratio < 3 else f'White Coating ({mildew_ratio:.1f}%)',
            'score': float(max(0, 100 - mildew_ratio * 10)),
            'confidence': min(96, 80 + (100 - mildew_ratio) * 0.16),
            'isHealthy': mildew_ratio < 5,
            'metrics': {'white_coverage': float(mildew_ratio)},
            'explanation': "White powdery coating from Podosphaera aphanis fungus.",
            'evidence': [f"White coverage: {mildew_ratio:.1f}%"],
            'recommendation': 'No powdery mildew detected.' if mildew_ratio < 5 else 
                            'Apply sulfur dust or potassium bicarbonate spray. Remove severely infected leaves.'
        }
        
        # 3. Anthracnose
        dark_only = cv2.bitwise_and(dark_mask, mask)
        dark_contours, _ = cv2.findContours(dark_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        anth_count = sum(1 for c in dark_contours if 100 < cv2.contourArea(c) < 3000)
        
        results['anthracnose'] = {
            'value': 'Not Detected' if anth_count < 2 else f'{anth_count} Lesion(s)',
            'score': float(max(0, 100 - anth_count * 15)),
            'confidence': min(93, 75 + (5 - min(anth_count, 5)) * 4),
            'isHealthy': anth_count < 3,
            'metrics': {'lesion_count': anth_count},
            'explanation': "Dark sunken lesions from Colletotrichum fungus.",
            'evidence': [f"Lesions detected: {anth_count}"],
            'recommendation': 'No anthracnose.' if anth_count < 3 else 
                            'Remove infected tissue. Apply copper or mancozeb fungicide.'
        }
        
        # 4. Angular Leaf Spot (Bacterial)
        angular_count = sum(1 for c in dark_contours if cv2.contourArea(c) > 50 and len(cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)) >= 4)
        
        results['angularSpot'] = {
            'value': 'Not Detected' if angular_count < 2 else f'{angular_count} Angular Spot(s)',
            'score': float(max(0, 100 - angular_count * 15)),
            'confidence': min(88, 72 + (5 - min(angular_count, 5)) * 4),
            'isHealthy': angular_count < 3,
            'metrics': {'angular_count': angular_count},
            'explanation': "Angular water-soaked spots from Xanthomonas bacteria.",
            'evidence': [f"Angular lesions: {angular_count}"],
            'recommendation': 'No bacterial spots.' if angular_count < 3 else 
                            'Apply copper bactericide. Avoid overhead irrigation.'
        }
        
        return results

    def analyze_nutrient_deficiency(self, image, mask, color_results):
        """Analyze key nutrient deficiencies"""
        results = {}
        
        chloro_score = color_results.get('chlorophyll', {}).get('score', 50)
        yellow_pct = color_results.get('chlorosis', {}).get('metrics', {}).get('yellow_percentage', 0)
        margin_pct = color_results.get('marginal', {}).get('metrics', {}).get('edge_browning_percentage', 0)
        interv_diff = color_results.get('interveinal', {}).get('metrics', {}).get('vein_interveinal_difference', 0)
        
        # 1. Nitrogen (Most Important)
        results['nitrogen'] = {
            'value': 'Adequate' if chloro_score > 60 else 'Deficient',
            'score': float(chloro_score),
            'confidence': min(93, 75 + chloro_score * 0.18),
            'isHealthy': chloro_score > 60,
            'metrics': {'chlorophyll_indicator': float(chloro_score)},
            'explanation': "Nitrogen assessed from overall leaf greenness.",
            'evidence': [f"Green color score: {chloro_score:.1f}/100"],
            'recommendation': 'Nitrogen levels adequate.' if chloro_score > 60 else 
                            'Apply nitrogen fertilizer: urea (46-0-0) or ammonium nitrate (34-0-0) at 0.5-1 lb N per 100 sq ft.'
        }
        
        # 2. Potassium (Second Priority)
        k_score = max(0, 100 - margin_pct * 4)
        results['potassium'] = {
            'value': 'Adequate' if margin_pct < 15 else 'Deficient',
            'score': float(k_score),
            'confidence': min(91, 74 + k_score * 0.17),
            'isHealthy': margin_pct < 15,
            'metrics': {'marginal_scorch': float(margin_pct)},
            'explanation': "Potassium deficiency causes brown leaf margins.",
            'evidence': [f"Edge browning: {margin_pct:.1f}%"],
            'recommendation': 'Potassium adequate.' if margin_pct < 15 else 
                            'Apply potassium sulfate (0-0-50) at 1-2 lbs per 100 sq ft.'
        }
        
        # 3. Iron (Third Priority)
        fe_score = max(0, 100 - interv_diff * 3)
        results['iron'] = {
            'value': 'Adequate' if interv_diff < 15 else 'Deficient',
            'score': float(fe_score),
            'confidence': min(89, 72 + fe_score * 0.17),
            'isHealthy': interv_diff < 15,
            'metrics': {'interveinal_index': float(interv_diff)},
            'explanation': "Iron deficiency shows as yellowing between green veins.",
            'evidence': [f"Interveinal chlorosis: {interv_diff:.1f}°"],
            'recommendation': 'Iron adequate.' if interv_diff < 15 else 
                            'Apply chelated iron (Fe-EDDHA) foliar spray or soil drench. Check soil pH (ideal 5.5-6.5).'
        }
        
        # 4. Magnesium (Fourth Priority)
        mg_score = max(0, 100 - interv_diff * 2 - yellow_pct * 0.3)
        results['magnesium'] = {
            'value': 'Adequate' if mg_score > 60 else 'Deficient',
            'score': float(mg_score),
            'confidence': min(88, 71 + mg_score * 0.17),
            'isHealthy': mg_score > 60,
            'metrics': {'combined_index': float(100 - mg_score)},
            'explanation': "Magnesium deficiency shows interveinal yellowing on older leaves.",
            'evidence': [f"Deficiency index: {100 - mg_score:.1f}"],
            'recommendation': 'Magnesium adequate.' if mg_score > 60 else 
                            'Apply Epsom salt (magnesium sulfate) at 2 tablespoons per gallon, foliar spray.'
        }
        
        return results

    def analyze_physical_damage(self, image, mask):
        """Analyze physical pest damage"""
        results = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Chewing Damage (Caterpillars, Beetles)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled_mask = np.zeros_like(mask)
            cv2.drawContours(filled_mask, contours, -1, 255, -1)
            holes = cv2.bitwise_and(filled_mask, cv2.bitwise_not(mask))
            hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hole_count = sum(1 for c in hole_contours if cv2.contourArea(c) > 30)
            hole_area = sum(cv2.contourArea(c) for c in hole_contours if cv2.contourArea(c) > 30)
            damage_pct = hole_area / max(1, np.sum(filled_mask > 0)) * 100
        else:
            hole_count, damage_pct = 0, 0
        
        results['chewing'] = {
            'value': 'None' if hole_count < 2 else f'{hole_count} Hole(s) - {"Light" if hole_count < 5 else "Moderate" if hole_count < 10 else "Severe"}',
            'score': float(max(0, 100 - hole_count * 15)),
            'confidence': min(96, 80 + (10 - min(hole_count, 10)) * 1.6),
            'isHealthy': hole_count < 3,
            'metrics': {'hole_count': hole_count, 'damage_percent': float(damage_pct)},
            'explanation': "Holes from caterpillars, beetles, or slugs.",
            'evidence': [f"Holes detected: {hole_count}", f"Damage: {damage_pct:.1f}%"],
            'recommendation': 'No chewing damage.' if hole_count < 3 else 
                            'Hand-pick pests at night. Apply spinosad or Bt (Bacillus thuringiensis) for caterpillars.'
        }
        
        # 2. Stippling (Spider Mites)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea, params.minArea, params.maxArea = True, 2, 50
        params.filterByCircularity, params.minCircularity = True, 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        keypoints = detector.detect(cv2.bitwise_not(gray_masked))
        dot_density = len(keypoints) / max(1, np.sum(mask > 0)) * 1000
        stip_sev = min(100, dot_density * 2)
        
        results['stippling'] = {
            'value': 'None' if stip_sev < 5 else 'Yellow Stippling Present',
            'score': float(max(0, 100 - stip_sev * 5)),
            'confidence': min(91, 75 + (100 - stip_sev) * 0.16),
            'isHealthy': stip_sev < 10,
            'metrics': {'dot_density': float(dot_density)},
            'explanation': "Tiny yellow dots from two-spotted spider mite feeding.",
            'evidence': [f"Stipple density: {dot_density:.1f} per 1000px²"],
            'recommendation': 'No mite damage.' if stip_sev < 10 else 
                            'Spray with water to dislodge mites. Apply insecticidal soap or horticultural oil.'
        }
        
        return results

    def analyze_environmental_stress(self, image, mask):
        """Analyze environmental stress"""
        results = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Sunburn/Heat Stress
        bleached = cv2.inRange(hsv, np.array([15, 20, 180]), np.array([35, 100, 255]))
        white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
        sun_mask = cv2.bitwise_and(cv2.bitwise_or(bleached, white), mask)
        sun_sev = np.sum(sun_mask > 0) / max(1, np.sum(mask > 0)) * 100
        
        results['sunburn'] = {
            'value': 'None' if sun_sev < 5 else 'Sunburn/Heat Damage',
            'score': float(max(0, 100 - sun_sev * 5)),
            'confidence': min(90, 74 + (100 - sun_sev) * 0.16),
            'isHealthy': sun_sev < 10,
            'metrics': {'bleached_percent': float(sun_sev)},
            'explanation': "Bleached areas from excessive sun exposure.",
            'evidence': [f"Bleached tissue: {sun_sev:.1f}%"],
            'recommendation': 'No sun damage.' if sun_sev < 10 else 
                            'Provide 30% shade cloth during heat waves. Ensure adequate irrigation.'
        }
        
        # 2. Water Stress
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            solidity = cv2.contourArea(cnt) / max(1, cv2.contourArea(hull))
        else:
            solidity = 1.0
        water_sev = max(0, (1 - solidity) * 200)
        
        results['waterStress'] = {
            'value': 'None' if water_sev < 10 else 'Wilting Detected',
            'score': float(max(0, 100 - water_sev * 4)),
            'confidence': min(90, 74 + (100 - water_sev) * 0.16),
            'isHealthy': water_sev < 15,
            'metrics': {'stress_severity': float(water_sev)},
            'explanation': "Leaf curling/wilting indicates water stress.",
            'evidence': [f"Wilt indicator: {water_sev:.1f}%"],
            'recommendation': 'Plant well-hydrated.' if water_sev < 15 else 
                            'Water deeply (1-1.5 inches per week). Check soil moisture at 4-6 inch depth.'
        }
        
        return results

    def analyze(self, image_path=None, base64_string=None):
        """Main analysis function - Streamlined"""
        try:
            image = self.load_image(image_path=image_path, base64_string=base64_string)
            image = self.preprocess_image(image)
            mask, contour = self.segment_leaf(image)
            
            if np.sum(mask > 0) < 1000:
                return {'success': False, 'error': 'Could not detect leaf. Upload clearer image.', 'results': {}}
            
            color_results = self.analyze_color_indicators(image, mask)
            structural_results = self.analyze_structural_features(image, mask, contour)
            disease_results = self.analyze_disease_symptoms(image, mask)
            nutrient_results = self.analyze_nutrient_deficiency(image, mask, color_results)
            physical_results = self.analyze_physical_damage(image, mask)
            environmental_results = self.analyze_environmental_stress(image, mask)
            
            all_results = {**color_results, **structural_results, **disease_results,
                          **nutrient_results, **physical_results, **environmental_results}
            
            scores = [r['score'] for r in all_results.values() if 'score' in r]
            overall_score = np.mean(scores) if scores else 0
            healthy_count = sum(1 for r in all_results.values() if r.get('isHealthy', False))
            
            issues = [{'attribute': k, 'value': v.get('value'), 'recommendation': v.get('recommendation')}
                     for k, v in all_results.items() if not v.get('isHealthy', True)]
            
            status = 'Healthy' if overall_score >= 80 else ('Good' if overall_score >= 60 else ('Moderate' if overall_score >= 40 else 'Poor'))
            
            return {
                'success': True,
                'overall_score': float(overall_score),
                'healthy_count': healthy_count,
                'total_count': len(all_results),
                'results': all_results,
                'summary': {
                    'status': status,
                    'issue_count': len(issues),
                    'issues': issues[:5],
                    'healthy_attributes': healthy_count
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'results': {}}


def main():
    if len(sys.argv) < 2:
        print("Usage: python strawberry_leaf_analyzer_trimmed.py <image_path>")
        print("       or: python strawberry_leaf_analyzer_trimmed.py --base64 < base64_data")
        sys.exit(1)
    
    analyzer = StrawberryLeafAnalyzer()
    
    if sys.argv[1] == '--base64':
        base64_data = sys.stdin.read().strip()
        results = analyzer.analyze(base64_string=base64_data)
    else:
        results = analyzer.analyze(image_path=sys.argv[1])
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results = convert_to_native(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
