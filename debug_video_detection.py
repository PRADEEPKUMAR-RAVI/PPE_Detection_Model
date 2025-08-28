#!/usr/bin/env python3
"""
Debug script to test video detection and see what's happening.
"""
import requests
import json

def test_video_detection():
    """Test video detection endpoint with a small video."""
    print("Testing video detection...")
    
    # Test with a sample video file (you'll need to provide a real video path)
    video_file_path = input("Enter path to test video file: ").strip()
    
    try:
        url = "http://localhost:8000/infer/video"
        
        with open(video_file_path, 'rb') as video_file:
            files = {"file": video_file}
            params = {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "save_outputs": False,
                "output_dir": "outputs",
                "download_video": False
            }
            
            print("Sending request to API...")
            response = requests.post(url, files=files, params=params, timeout=300)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"People detected: {result.get('counts', {}).get('total', 0)}")
                print(f"Safe: {result.get('counts', {}).get('safe', 0)}")
                print(f"Unsafe: {result.get('counts', {}).get('unsafe', 0)}")
                
                if 'analytics' in result:
                    analytics = result['analytics']
                    print(f"Unique people: {analytics.get('unique_people_detected', 0)}")
                    print(f"Total detections: {analytics.get('total_person_detections', 0)}")
                    print(f"Frames with detections: {analytics.get('frames_with_detections', 0)}")
                
                if 'people' in result:
                    print(f"People details:")
                    for person in result['people']:
                        print(f"  Person {person.get('person_id', '?')}: {person.get('status', '?')}")
                        if 'tracking_stats' in person:
                            stats = person['tracking_stats']
                            print(f"    Appearances: {stats.get('total_appearances', 0)}")
                            print(f"    Safety %: {stats.get('safety_percentage', 0):.1f}%")
                
            else:
                print(f"❌ Error: {response.text}")
                
    except FileNotFoundError:
        print("❌ Video file not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_video_detection()