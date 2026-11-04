// Send the File directly to the session's HTTP endpoint. Only its one-time token crosses trame.
(function () {
  "use strict";

  const projectUpload = {
    props: {
      disabled: Boolean,
      result: Object,
      limit: Number,
    },
    emits: ["uploaded"],
    data() {
      return { busy: false, percent: null, status: "", failed: false, token: null };
    },
    watch: {
      result(response) {
        if (response && this.token && response.token === this.token) {
          this.busy = false;
          this.token = null;
          this.failed = !response.ok;
          this.status = response.message;
        }
      },
    },
    methods: {
      setError(message) {
        this.busy = false;
        this.percent = null;
        this.failed = true;
        this.status = message;
      },
      upload(event) {
        const file = event.target.files[0];
        event.target.value = "";
        if (!file || this.busy || this.disabled) return;
        if (file.size > this.limit) {
          this.setError(`File exceeds the ${this.limit / 1024 ** 2} MiB upload limit.`);
          return;
        }
        this.busy = true;
        this.failed = false;
        this.token = null;
        this.percent = 0;
        this.status = `Uploading ${file.name}`;
        const data = new FormData();
        data.append("file", file);
        const request = new XMLHttpRequest();
        request.open("POST", new URL("upload/project", document.baseURI));
        request.responseType = "json";
        request.upload.onprogress = (progress) => {
          this.percent = progress.lengthComputable ? Math.round(progress.loaded * 100 / progress.total) : null;
        };
        request.onload = () => {
          const response = request.response;
          if (request.status !== 200 || !response || !response.token) {
            this.setError(response?.error || `Upload failed (HTTP ${request.status}). Check your workspace login and retry.`);
            return;
          }
          this.percent = 100;
          this.status = `Upload complete. Opening ${response.name}…`;
          this.token = response.token;
          this.$emit("uploaded", this.token);
        };
        request.onerror = () => this.setError("Upload interrupted. Check the connection and select the file again.");
        request.onabort = () => this.setError("Upload canceled.");
        request.send(data);
      },
    },
    template: `
      <div>
        <label for="coilgen-project-file" class="text-caption">Open project</label>
        <input id="coilgen-project-file" type="file" accept=".coilgen,.json" :disabled="disabled || busy"
               style="display:block;max-width:100%" @change="upload">
        <small v-if="!status">Compact .coilgen or project JSON</small>
        <div v-if="busy"><progress max="100" :value="percent" aria-label="Project upload progress"></progress>
          <span v-if="percent !== null">{{ percent }}%</span></div>
        <div role="status" aria-live="polite" class="text-caption" :class="{ 'text-error': failed }">{{ status }}</div>
      </div>`,
  };

  window.CoilGenProjectUpload = {
    install(app) { app.component("coilgen-project-upload", projectUpload); },
  };
})();
