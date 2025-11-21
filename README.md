# THUACP-ML-25
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

warnings.filterwarnings('ignore')

# ===================== 0. Setup Logging =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 1. Configuration Parameters =====================
class Config:
    """Centralized configuration management"""
    
    # Root directory of your dataset
    DATA_ROOT = "./"  # Local: path to dataset; Kaggle: '/kaggle/input/MABe-mouse-behavior-detection/'
    
    # Fixed file paths
    METADATA_TRAIN = None  # Will be set in __init__
    METADATA_TEST = None
    TRACKING_TRAIN_DIR = None
    TRACKING_TEST_DIR = None
    ANNOTATIONS_TRAIN = None
    SAMPLE_SUBMISSION = None
    OUTPUT_DIR = None
    
    # Preprocessing parameters
    MIN_FRAMES_PER_BEHAVIOR = 5  # Filter behaviors shorter than N frames
    KEY_BODYPARTS = ["center", "bodycenter", "head", "nose", "tail", "tailbase"]
    INTERPOLATE_LIMIT = 3  # Max consecutive missing frames for interpolation
    
    # Feature engineering parameters
    VELOCITY_WINDOW = 3  # Frames for velocity calculation
    ACCELERATION_WINDOW = 3  # Frames for acceleration calculation
    RELATIVE_DISTANCE_THRESHOLD = 50  # cm threshold for social interactions
    
    def __init__(self, data_root: str = None):
        """Initialize paths based on data_root"""
        if data_root:
            self.DATA_ROOT = data_root.rstrip('/') + '/'
        
        self.METADATA_TRAIN = f"{self.DATA_ROOT}train.csv"
        self.METADATA_TEST = f"{self.DATA_ROOT}test.csv"
        self.TRACKING_TRAIN_DIR = f"{self.DATA_ROOT}train_tracking/"
        self.TRACKING_TEST_DIR = f"{self.DATA_ROOT}test_tracking/"
        self.ANNOTATIONS_TRAIN = f"{self.DATA_ROOT}train_annotation.csv"
        self.SAMPLE_SUBMISSION = f"{self.DATA_ROOT}sample_submission.csv"
        self.OUTPUT_DIR = f"{self.DATA_ROOT}processed_data/"
        
        # Create output directory
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def validate_paths(self, split: str = "train") -> bool:
        """Validate that required files exist"""
        required_paths = [
            self.METADATA_TRAIN if split == "train" else self.METADATA_TEST,
            self.TRACKING_TRAIN_DIR if split == "train" else self.TRACKING_TEST_DIR,
        ]
        if split == "train":
            required_paths.append(self.ANNOTATIONS_TRAIN)
        
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            logger.error(f"Missing required files: {missing}")
            return False
        return True

# ===================== 2. Load Metadata =====================
def load_metadata(config: Config, split: str = "train") -> Dict:
    """Load metadata for train/test set, return dict with video_id as key"""
    metadata_path = config.METADATA_TRAIN if split == "train" else config.METADATA_TEST
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        metadata = pd.read_csv(metadata_path)
        
        # Validate required columns
        required_cols = ["video_id", "lab_id"]
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in metadata: {missing_cols}")
        
        # Retain and rename key metadata fields
        key_cols = [
            "video_id", "lab_id", "frames per second", "pix per cm (approx)",
            "arena width (cm)", "arena height (cm)", "video width", "video height"
        ]
        available_cols = [col for col in key_cols if col in metadata.columns]
        metadata = metadata[available_cols].copy()
        
        rename_map = {
            "frames per second": "fps",
            "pix per cm (approx)": "pix_per_cm",
            "arena width (cm)": "arena_w_cm",
            "arena height (cm)": "arena_h_cm",
            "video width": "video_w",
            "video height": "video_h"
        }
        metadata = metadata.rename(columns={col: rename_map.get(col, col) for col in metadata.columns})
        
        # Fill missing pix_per_cm with lab-specific average
        if "pix_per_cm" in metadata.columns:
            metadata["pix_per_cm"] = metadata.groupby("lab_id")["pix_per_cm"].transform(
                lambda x: x.fillna(x.mean()) if x.notna().any() else x
            )
            # If still missing, use global mean
            metadata["pix_per_cm"] = metadata["pix_per_cm"].fillna(metadata["pix_per_cm"].mean())
        
        # Fill missing arena dimensions with defaults
        if "arena_w_cm" in metadata.columns:
            metadata["arena_w_cm"] = metadata["arena_w_cm"].fillna(metadata["arena_w_cm"].median())
        if "arena_h_cm" in metadata.columns:
            metadata["arena_h_cm"] = metadata["arena_h_cm"].fillna(metadata["arena_h_cm"].median())
        
        # Convert to dict
        metadata_dict = metadata.set_index("video_id").to_dict("index")
        logger.info(f"Loaded {split} set metadata: {len(metadata_dict)} videos")
        
        return metadata_dict
    
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise

# ===================== 3. Load Tracking Data =====================
def load_tracking_data(video_id: str, config: Config, split: str = "train") -> Optional[pd.DataFrame]:
    """Load tracking data for a single video, restructure to 'frame+mouse+bodypart' format"""
    tracking_dir = config.TRACKING_TRAIN_DIR if split == "train" else config.TRACKING_TEST_DIR
    tracking_path = f"{tracking_dir}{video_id}.csv"
    
    if not os.path.exists(tracking_path):
        logger.warning(f"Tracking file for {video_id} not found: {tracking_path}")
        return None
    
    try:
        tracking = pd.read_csv(tracking_path)
        
        # Standardize column names
        column_mapping = {}
        if "video_frame" in tracking.columns:
            column_mapping["video_frame"] = "frame"
        if "mouse_id" in tracking.columns:
            column_mapping["mouse_id"] = "agent_id"
        if "x" in tracking.columns:
            column_mapping["x"] = "x_pix"
        if "y" in tracking.columns:
            column_mapping["y"] = "y_pix"
        
        tracking = tracking.rename(columns=column_mapping)
        
        # Validate required columns
        required_cols = ["frame", "agent_id", "bodypart", "x_pix", "y_pix"]
        missing_cols = [col for col in required_cols if col not in tracking.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {video_id}: {missing_cols}")
            return None
        
        # Normalize bodypart names
        tracking["bodypart"] = tracking["bodypart"].str.lower().str.strip()
        
        # Filter key body parts
        valid_bodyparts = [bp for bp in config.KEY_BODYPARTS if bp in tracking["bodypart"].unique()]
        if not valid_bodyparts:
            logger.warning(f"No valid body part data for {video_id}, skipping")
            return None
        
        tracking = tracking[tracking["bodypart"].isin(valid_bodyparts)].copy()
        
        # Remove duplicates
        tracking = tracking.drop_duplicates(subset=["frame", "agent_id", "bodypart"])
        
        # Remove invalid coordinates (NaN or extreme values)
        tracking = tracking[
            tracking["x_pix"].notna() & tracking["y_pix"].notna() &
            (tracking["x_pix"].abs() < 1e6) & (tracking["y_pix"].abs() < 1e6)
        ]
        
        if len(tracking) == 0:
            logger.warning(f"No valid tracking data for {video_id}")
            return None
        
        # Pivot body parts to columns
        tracking_pivoted = tracking.pivot_table(
            index=["frame", "agent_id"],
            columns="bodypart",
            values=["x_pix", "y_pix"],
            aggfunc="first"
        ).reset_index()
        
        # Flatten column names
        tracking_pivoted.columns = ["_".join(str(col).strip("_") for col in col_tuple).strip("_") 
                                    if isinstance(col_tuple, tuple) else str(col_tuple)
                                    for col_tuple in tracking_pivoted.columns]
        
        # Ensure column names are strings
        tracking_pivoted.columns = [str(col) for col in tracking_pivoted.columns]
        
        # Add missing body part columns
        for bp in valid_bodyparts:
            for coord in ["x_pix", "y_pix"]:
                col = f"{coord}_{bp}"
                if col not in tracking_pivoted.columns:
                    tracking_pivoted[col] = np.nan
        
        num_mice = tracking['agent_id'].nunique()
        logger.debug(f"Video {video_id}: Loaded {len(tracking_pivoted)} records ({num_mice} mice)")
        
        return tracking_pivoted
    
    except Exception as e:
        logger.error(f"Error loading tracking data for {video_id}: {e}")
        return None

# ===================== 4. Load Annotations =====================
def load_annotations(config: Config) -> Dict:
    """Load training set annotations, return dict grouped by video_id"""
    if not os.path.exists(config.ANNOTATIONS_TRAIN):
        raise FileNotFoundError(f"Annotation file not found: {config.ANNOTATIONS_TRAIN}")
    
    try:
        annotations = pd.read_csv(config.ANNOTATIONS_TRAIN)
        
        # Validate required columns
        required_cols = ["video_id", "agent_id", "action", "start_frame", "stop_frame"]
        missing_cols = [col for col in required_cols if col not in annotations.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in annotations: {missing_cols}")
        
        # Filter short behaviors
        annotations["duration_frames"] = annotations["stop_frame"] - annotations["start_frame"] + 1
        initial_count = len(annotations)
        annotations = annotations[annotations["duration_frames"] >= config.MIN_FRAMES_PER_BEHAVIOR].copy()
        filtered_count = initial_count - len(annotations)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} short behaviors (< {config.MIN_FRAMES_PER_BEHAVIOR} frames)")
        
        # Normalize action labels
        annotations["action"] = annotations["action"].str.lower().str.replace(" ", "_").str.strip()
        
        # Group by video_id
        annotations_dict = {
            video_id: group.drop("duration_frames", axis=1)
            for video_id, group in annotations.groupby("video_id")
        }
        
        logger.info(f"Loaded training annotations: {len(annotations_dict)} videos, "
                   f"{len(annotations)} behavior records total")
        
        return annotations_dict
    
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        raise

# ===================== 5. Feature Engineering =====================
def calculate_velocity_features(df: pd.DataFrame, fps: float, window: int = 3) -> pd.DataFrame:
    """Calculate velocity and speed features"""
    df = df.copy()
    df = df.sort_values(["agent_id", "frame"])
    
    for agent_id in df["agent_id"].unique():
        mask = df["agent_id"] == agent_id
        agent_data = df.loc[mask].copy()
        
        # Calculate displacement
        dx = agent_data["x_cm"].diff(window)
        dy = agent_data["y_cm"].diff(window)
        
        # Calculate velocity (cm/s)
        dt = window / fps
        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        # Update dataframe
        df.loc[mask, "velocity_x"] = vx.values
        df.loc[mask, "velocity_y"] = vy.values
        df.loc[mask, "speed"] = speed.values
    
    # Fill NaN values (first few frames)
    df[["velocity_x", "velocity_y", "speed"]] = df[["velocity_x", "velocity_y", "speed"]].fillna(0)
    
    return df

def calculate_acceleration_features(df: pd.DataFrame, fps: float, window: int = 3) -> pd.DataFrame:
    """Calculate acceleration features"""
    df = df.copy()
    df = df.sort_values(["agent_id", "frame"])
    
    for agent_id in df["agent_id"].unique():
        mask = df["agent_id"] == agent_id
        agent_data = df.loc[mask].copy()
        
        # Calculate acceleration from velocity
        dt = window / fps
        ax = agent_data["velocity_x"].diff(window) / dt
        ay = agent_data["velocity_y"].diff(window) / dt
        acceleration = np.sqrt(ax**2 + ay**2)
        
        df.loc[mask, "acceleration_x"] = ax.values
        df.loc[mask, "acceleration_y"] = ay.values
        df.loc[mask, "acceleration"] = acceleration.values
    
    df[["acceleration_x", "acceleration_y", "acceleration"]] = \
        df[["acceleration_x", "acceleration_y", "acceleration"]].fillna(0)
    
    return df

def calculate_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative positions and distances between mice in the same video"""
    df = df.copy()
    
    # Initialize relative features
    df["nearest_mouse_distance"] = np.nan
    df["nearest_mouse_id"] = np.nan
    df["num_mice_in_range"] = 0
    
    for video_id in df["video_id"].unique():
        video_data = df[df["video_id"] == video_id].copy()
        
        for frame in video_data["frame"].unique():
            frame_data = video_data[video_data["frame"] == frame].copy()
            
            if len(frame_data) < 2:
                continue
            
            # Calculate pairwise distances
            positions = frame_data[["agent_id", "x_cm", "y_cm"]].values
            for i, (aid1, x1, y1) in enumerate(positions):
                distances = []
                other_ids = []
                
                for j, (aid2, x2, y2) in enumerate(positions):
                    if aid1 != aid2:
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        distances.append(dist)
                        other_ids.append(aid2)
                
                if distances:
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    nearest_id = other_ids[min_idx]
                    num_in_range = sum(1 for d in distances if d < 50)  # 50cm threshold
                    
                    # Update dataframe
                    mask = (df["video_id"] == video_id) & (df["frame"] == frame) & (df["agent_id"] == aid1)
                    df.loc[mask, "nearest_mouse_distance"] = min_dist
                    df.loc[mask, "nearest_mouse_id"] = nearest_id
                    df.loc[mask, "num_mice_in_range"] = num_in_range
    
    return df

# ===================== 6. Core Preprocessing =====================
def preprocess_single_video(tracking_data: pd.DataFrame, video_metadata: Dict, 
                           config: Config) -> Optional[pd.DataFrame]:
    """
    Preprocess single video data with enhanced features
    """
    if tracking_data is None or video_metadata is None:
        return None
    
    try:
        processed = tracking_data.copy()
        pix_per_cm = video_metadata.get("pix_per_cm", 1.0)
        fps = video_metadata.get("fps", 30.0)
        arena_w = video_metadata.get("arena_w_cm", 50.0)
        arena_h = video_metadata.get("arena_h_cm", 50.0)
        
        if pix_per_cm <= 0 or fps <= 0:
            logger.warning(f"Invalid metadata: pix_per_cm={pix_per_cm}, fps={fps}")
            return None
        
        # Step 1: Pixel → cm Conversion
        for col in processed.columns:
            if col.startswith("x_pix_"):
                bp = col.replace("x_pix_", "")
                processed[f"x_cm_{bp}"] = processed[col] / pix_per_cm
            elif col.startswith("y_pix_"):
                bp = col.replace("y_pix_", "")
                processed[f"y_cm_{bp}"] = processed[col] / pix_per_cm
        
        # Step 2: Extract Core Coordinates
        center_x_cols = [col for col in processed.columns 
                        if col.startswith("x_cm_") and any(bp in col for bp in ["center", "bodycenter"])]
        center_y_cols = [col for col in processed.columns 
                        if col.startswith("y_cm_") and any(bp in col for bp in ["center", "bodycenter"])]
        
        if not center_x_cols:
            center_x_cols = [col for col in processed.columns if "x_cm_head" in col]
            center_y_cols = [col for col in processed.columns if "y_cm_head" in col]
        
        if center_x_cols and center_y_cols:
            processed["x_cm"] = processed[center_x_cols[0]]
            processed["y_cm"] = processed[center_y_cols[0]]
        else:
            logger.warning("No valid center coordinates found")
            return None
        
        # Step 3: Frame Alignment + Interpolation
        processed = processed.sort_values(["agent_id", "frame"])
        processed_list = []
        
        for agent_id in processed["agent_id"].unique():
            agent_data = processed[processed["agent_id"] == agent_id].copy()
            
            # Generate continuous frame numbers
            min_frame = agent_data["frame"].min()
            max_frame = agent_data["frame"].max()
            full_frames = pd.DataFrame({"frame": range(int(min_frame), int(max_frame) + 1)})
            agent_data = full_frames.merge(agent_data, on="frame", how="left")
            agent_data["agent_id"] = agent_id
            
            # Temporal interpolation
            coord_cols = [col for col in agent_data.columns if col.startswith(("x_cm", "y_cm"))]
            agent_data[coord_cols] = agent_data[coord_cols].interpolate(
                method="linear",
                limit=config.INTERPOLATE_LIMIT,
                limit_direction="both"
            )
            
            # Fill remaining NaN with forward/backward fill, then median
            agent_data = agent_data.ffill().bfill()
            numeric_cols = agent_data.select_dtypes(include=[np.number]).columns
            agent_data[numeric_cols] = agent_data[numeric_cols].fillna(agent_data[numeric_cols].median())
            
            processed_list.append(agent_data)
        
        processed = pd.concat(processed_list, ignore_index=True)
        
        # Step 4: Basic Feature Engineering
        processed["time_sec"] = processed["frame"] / fps
        
        # Orientation
        if "x_cm_head" in processed.columns and "y_cm_head" in processed.columns:
            dx = processed["x_cm_head"] - processed["x_cm"]
            dy = processed["y_cm_head"] - processed["y_cm"]
            processed["orientation_rad"] = np.arctan2(dy, dx)
            processed["orientation_deg"] = np.degrees(processed["orientation_rad"])
        
        # Body length
        if all(col in processed.columns for col in ["x_cm_head", "y_cm_head", "x_cm_tail", "y_cm_tail"]):
            processed["body_length_cm"] = np.sqrt(
                (processed["x_cm_head"] - processed["x_cm_tail"])**2 +
                (processed["y_cm_head"] - processed["y_cm_tail"])**2
            )
        
        # Normalized coordinates
        processed["x_norm"] = processed["x_cm"] / arena_w
        processed["y_norm"] = processed["y_cm"] / arena_h
        
        # Step 5: Advanced Features (Velocity, Acceleration)
        processed = calculate_velocity_features(processed, fps, config.VELOCITY_WINDOW)
        processed = calculate_acceleration_features(processed, fps, config.ACCELERATION_WINDOW)
        
        # Step 6: Relative Features (for social behaviors)
        # Note: This can be computationally expensive, consider making it optional
        # processed = calculate_relative_features(processed)
        
        # Step 7: Filter Invalid Data
        processed = processed.dropna(subset=["x_cm", "y_cm"])
        processed = processed.reset_index(drop=True)
        
        if len(processed) == 0:
            logger.warning("No valid data after preprocessing")
            return None
        
        return processed
    
    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ===================== 7. Batch Preprocessing =====================
def batch_preprocess(config: Config, split: str = "train", 
                    max_videos: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Batch process all videos, return merged preprocessed data + annotations"""
    
    # Validate paths
    if not config.validate_paths(split):
        raise ValueError(f"Path validation failed for {split} set")
    
    # Load metadata
    metadata_dict = load_metadata(config, split)
    annotations_dict = load_annotations(config) if split == "train" else None
    
    # Limit number of videos for testing
    if max_videos:
        video_ids = list(metadata_dict.keys())[:max_videos]
        metadata_dict = {vid: metadata_dict[vid] for vid in video_ids}
        logger.info(f"Processing limited to {max_videos} videos")
    
    all_processed_poses = []
    all_annotations = []
    failed_videos = []
    
    # Process videos
    for video_id, video_metadata in tqdm(metadata_dict.items(), desc=f"Processing {split} set"):
        try:
            # Load tracking data
            tracking_data = load_tracking_data(video_id, config, split)
            if tracking_data is None:
                failed_videos.append(video_id)
                continue
            
            # Preprocess
            processed_poses = preprocess_single_video(tracking_data, video_metadata, config)
            if processed_poses is None:
                failed_videos.append(video_id)
                continue
            
            # Add metadata
            processed_poses["video_id"] = video_id
            processed_poses["lab_id"] = video_metadata.get("lab_id", "unknown")
            
            all_processed_poses.append(processed_poses)
            
            # Add annotations for training set
            if split == "train" and annotations_dict and video_id in annotations_dict:
                annotations = annotations_dict[video_id].copy()
                annotations["lab_id"] = video_metadata.get("lab_id", "unknown")
                all_annotations.append(annotations)
        
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            failed_videos.append(video_id)
            continue
    
    # Merge all data
    if not all_processed_poses:
        raise ValueError("No videos were successfully processed!")
    
    merged_poses = pd.concat(all_processed_poses, ignore_index=True)
    merged_annotations = pd.concat(all_annotations, ignore_index=True) if all_annotations else None
    
    # Save results
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    output_poses_path = f"{config.OUTPUT_DIR}{split}_processed_poses.csv"
    merged_poses.to_csv(output_poses_path, index=False)
    logger.info(f"Saved processed poses to: {output_poses_path}")
    
    if split == "train" and merged_annotations is not None:
        output_annotations_path = f"{config.OUTPUT_DIR}train_processed_annotations.csv"
        merged_annotations.to_csv(output_annotations_path, index=False)
        logger.info(f"Saved annotations to: {output_annotations_path}")
    
    # Save processing statistics
    stats = {
        "split": split,
        "total_videos": len(metadata_dict),
        "successful_videos": len(metadata_dict) - len(failed_videos),
        "failed_videos": len(failed_videos),
        "failed_video_ids": failed_videos,
        "total_pose_records": len(merged_poses),
        "unique_videos": merged_poses["video_id"].nunique() if "video_id" in merged_poses.columns else 0,
        "unique_mice": merged_poses["agent_id"].nunique() if "agent_id" in merged_poses.columns else 0,
    }
    
    if split == "train" and merged_annotations is not None:
        stats["total_annotations"] = len(merged_annotations)
        stats["unique_actions"] = merged_annotations["action"].nunique()
        stats["action_distribution"] = merged_annotations["action"].value_counts().to_dict()
    
    stats_path = f"{config.OUTPUT_DIR}{split}_processing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved processing statistics to: {stats_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{split.upper()} SET PREPROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"Videos processed: {stats['successful_videos']}/{stats['total_videos']}")
    print(f"Failed videos: {len(failed_videos)}")
    print(f"Pose records: {stats['total_pose_records']:,}")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"Unique mice: {stats['unique_mice']}")
    
    if split == "train" and merged_annotations is not None:
        print(f"Annotation records: {stats['total_annotations']:,}")
        print(f"Unique actions: {stats['unique_actions']}")
        print(f"\nTop 10 Actions:")
        for action, count in list(stats["action_distribution"].items())[:10]:
            print(f"  {action}: {count}")
    
    print(f"\nOutput directory: {config.OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    return merged_poses, merged_annotations

# ===================== 8. Main Execution =====================
if __name__ == "__main__":
    # Initialize configuration
    config = Config(data_root="./")  # Adjust path as needed
    
    # Process training set
    logger.info("Starting training set preprocessing...")
    train_poses, train_annotations = batch_preprocess(config, split="train")
    
    # Optional: Process test set
    # logger.info("Starting test set preprocessing...")
    # test_poses, _ = batch_preprocess(config, split="test")
    
    logger.info("Preprocessing completed successfully!")
    
