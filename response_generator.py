class ResponseGenerator:
    @staticmethod
    def generate(row):
        print(ResponseGenerator.render(row))

    @staticmethod
    def render(row):
        output = []
        output.append(f"\n🚨 Emergency Detected: {row.get('Emergency Type', 'Unknown')}")

        if row.get("Symptoms"):
            output.append("\n🩺 Symptoms to Watch For:")
            for line in row["Symptoms"].split("\n"):
                output.append(f"- {line.strip()}")

        if row.get("When to Call Emergency Services"):
            output.append("\n📞 Call Emergency Services If:")
            for line in row["When to Call Emergency Services"].split("\n"):
                output.append(f"- {line.strip()}")

        if row.get("Immediate Actions"):
            output.append("\n✅ Immediate First Aid Steps:")
            for idx, step in enumerate(row["Immediate Actions"].split("\n"), 1):
                output.append(f"{idx}. {step.strip()}")

        if row.get("Do's and Don'ts"):
            output.append("\n⚠️ Do's and Don'ts:")
            for line in row["Do's and Don'ts"].split("\n"):
                output.append(f"- {line.strip()}")

        if row.get("Follow-Up Instructions"):
            output.append("\n🔄 Follow-Up Instructions:")
            for line in row["Follow-Up Instructions"].split("\n"):
                output.append(f"- {line.strip()}")

        if row.get("Severity Level"):
            sev = row["Severity Level"].strip().capitalize()
            severity_emoji = "🟢 Mild" if "mild" in sev.lower() else "🟠 Moderate" if "moderate" in sev.lower() else "🔴 Severe"
            output.append(f"\n📊 Severity Level: {severity_emoji}")

        output.append("\n📌 Stay calm and follow these steps. This information is for guidance only. Seek professional help when needed.")
        return "\n".join(output)

