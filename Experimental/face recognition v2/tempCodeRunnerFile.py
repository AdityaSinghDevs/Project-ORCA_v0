    max_score = 0.0
        sim_user_id = ""
        
        # Debug prints
        print("Real-time feature shape:", feature1.shape)
        
        for user_id, feature2 in self.dictionary.items():
            print(f"Stored feature shape for {user_id}:", feature2.shape)
            # Make sure feature2 is the right type and shape
            feature2 = np.array(feature2, dtype=np.float32)
            
            try:
                score = self.face_recognizer.match(
                    feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
                if score >= max_score:
                    max_score = score
                    sim_user_id = user_id
            except cv2.error as e:
                print(f"Error matching with {user_id}:", e)
                continue
                    
        if max_score < self.cosine_threshold:
            return False, ("", 0.0)
        return True, (sim_user_id, max_score)