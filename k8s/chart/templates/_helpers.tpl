{{- define "tutor.namespace" -}}
lab-{{ .Values.participant }}
{{- end -}}

{{- define "tutor.labels" -}}
app.kubernetes.io/part-of: sign-tutor
app.kubernetes.io/instance: {{ .Values.participant }}
{{- end -}}
