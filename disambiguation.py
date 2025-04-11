import logging
import os
import csv

class Disambiguator:
    def __init__(self, config):
        self.threshold = float(config['disambiguation_threshold'])
        self.log_path = "logs/disambiguation_log.csv"
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["user_input", "choice", "top1", "top2", "top3", "score_gap", "top_score"])

    def resolve(self, matches, user_input, auto=True):
        if len(matches) < 2:
            logging.info("Disambiguation skipped: Only one match available.")
            return matches.iloc[0]

        top_score = matches.iloc[0]["Score"]
        second_score = matches.iloc[1]["Score"]
        score_gap = top_score - second_score
        use_top_3 = top_score < 0.7 or score_gap < self.threshold

        if not use_top_3:
            logging.info(f"High confidence (score={top_score:.2f}) — auto-selected top match.")
            return matches.iloc[0]

        top3 = matches.head(3)

        if auto:
            # In API mode, do NOT use input(), just trigger disambiguation UI
            logging.info("Disambiguation triggered in API mode.")
            self.log_disambiguation(user_input, "API_Disambiguation",
                                    top3.iloc[0]["Emergency Type"],
                                    top3.iloc[1]["Emergency Type"] if len(top3) > 1 else "",
                                    top3.iloc[2]["Emergency Type"] if len(top3) > 2 else "",
                                    score_gap, top_score)
            return None

        # CLI Mode (run()) — user can select from numbered options
        print("⚠️ Multiple possible emergencies detected. Please choose the best match:")
        for i, row in top3.iterrows():
            print(f"{i + 1}. {row['Emergency Type']}")
        print(f"{len(top3) + 1}. None of these – I’ll try again if you describe it differently.")

        choice = input(f"Select 1-{len(top3)+1}: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(top3):
                selected = top3.iloc[idx - 1]
                self.log_disambiguation(user_input, selected["Emergency Type"],
                                        top3.iloc[0]["Emergency Type"],
                                        top3.iloc[1]["Emergency Type"] if len(top3) > 1 else "",
                                        top3.iloc[2]["Emergency Type"] if len(top3) > 2 else "",
                                        score_gap, top_score)
                return selected
            else:
                self.log_disambiguation(user_input, "None",
                                        top3.iloc[0]["Emergency Type"],
                                        top3.iloc[1]["Emergency Type"] if len(top3) > 1 else "",
                                        top3.iloc[2]["Emergency Type"] if len(top3) > 2 else "",
                                        score_gap, top_score)
                return None
        except:
            print("Invalid input. Skipping disambiguation.")
            self.log_disambiguation(user_input, "Invalid",
                                    top3.iloc[0]["Emergency Type"],
                                    top3.iloc[1]["Emergency Type"] if len(top3) > 1 else "",
                                    top3.iloc[2]["Emergency Type"] if len(top3) > 2 else "",
                                    score_gap, top_score)
            return None

    def log_disambiguation(self, user_input, choice, top1, top2, top3, score_gap, top_score):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                user_input, choice, top1, top2, top3, f"{score_gap:.4f}", f"{top_score:.4f}"
            ])
