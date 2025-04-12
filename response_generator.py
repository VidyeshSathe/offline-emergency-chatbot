class ResponseGenerator:
    @staticmethod
    def render(row):
        response = {
            "emergency": row.get("Emergency Type", "Unknown"),
            "symptoms": [],
            "call_emergency_if": [],
            "first_aid_steps": [],
            "dos_and_donts": [],
            "follow_up": [],
            "severity": "Unknown",
            "severity_emoji": "❓",
            "disclaimer": (
                "📌 Stay calm and follow these steps. This information is for guidance only. "
                "Seek professional help when needed."
            )
        }

        if row.get("Symptoms"):
            response["symptoms"] = [line.strip() for line in row["Symptoms"].split("\n") if line.strip()]

        if row.get("When to Call Emergency Services"):
            response["call_emergency_if"] = [line.strip() for line in row["When to Call Emergency Services"].split("\n") if line.strip()]

        if row.get("Immediate Actions"):
            response["first_aid_steps"] = [line.strip() for line in row["Immediate Actions"].split("\n") if line.strip()]

        if row.get("Do's and Don'ts"):
            response["dos_and_donts"] = [line.strip() for line in row["Do's and Don'ts"].split("\n") if line.strip()]

        if row.get("Follow-Up Instructions"):
            response["follow_up"] = [line.strip() for line in row["Follow-Up Instructions"].split("\n") if line.strip()]

        if row.get("Severity Level"):
            level = row["Severity Level"].strip().lower()
            response["severity"] = level.capitalize()
            response["severity_emoji"] = (
                "🟢" if "mild" in level else "🟠" if "moderate" in level else "🔴" if "severe" in level else "❓"
            )

        return response


