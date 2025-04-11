class ResponseGenerator:
    @staticmethod
    def generate(row):
        print(ResponseGenerator.render(row))

    @staticmethod
    def render(row):
        output = []
        output.append(f"🚨 Emergency Detected: {row.get('Emergency Type', 'Unknown')}\n")

        # Symptoms
        symptoms = row.get("Symptoms")
        if symptoms:
            output.append("🩺 Symptoms to Watch For:")
            for line in symptoms.split("\n"):
                line = line.strip()
                if line:
                    output.append(f"- {line}")
            output.append("")

        # When to Call
        when = row.get("When to Call Emergency Services")
        if when:
            output.append("📞 Call Emergency Services If:")
            for line in when.split("\n"):
                line = line.strip()
                if line:
                    output.append(f"- {line}")
            output.append("")

        # Immediate Actions
        actions = row.get("Immediate Actions")
        if actions:
            output.append("✅ Immediate First Aid Steps:")
            for i, step in enumerate(actions.split("\n"), 1):
                step = step.strip().lstrip("0123456789. ").strip()
                if step:
                    output.append(f"{i}. {step}")
            output.append("")

        # Do's and Don'ts
        dos = row.get("Do's and Don'ts")
        if dos:
            output.append("⚠️ Do's and Don'ts:")
            for line in dos.split("\n"):
                line = line.strip()
                if line:
                    output.append(f"- {line}")
            output.append("")

        # Follow-Up
        follow = row.get("Follow-Up Instructions")
        if follow:
            output.append("🔄 Follow-Up Instructions:")
            for line in follow.split("\n"):
                line = line.strip()
                if line:
                    output.append(f"- {line}")
            output.append("")

        # Severity
        sev = row.get("Severity Level", "").strip().lower()
        if sev:
            severity = "🟢 Mild" if "mild" in sev else "🟠 Moderate" if "moderate" in sev else "🔴 Severe"
            output.append(f"📊 Severity Level: {severity}")

        output.append("\n📌 Stay calm and follow these steps. This information is for guidance only. Seek professional help when needed.")

        return "\n".join(output)

