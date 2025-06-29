<!-- src/components/AgentInterface.vue -->
<template>
  <div class="container">
    <h1>Assistant IA RAG</h1>

    <textarea v-model="input" placeholder="Entrez votre texte ici..."></textarea>

    <div class="buttons">
      <button @click="embed">Embed</button>
      <button @click="search">Rechercher</button>
      <button @click="askAgent">Poser à l'agent</button>
    </div>

    <div v-if="response">
      <h3>Résultat :</h3>
      <pre>{{ response }}</pre>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      input: "",
      response: "",
    };
  },
  methods: {
    async embed() {
      const res = await fetch("http://localhost:5000/embed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: [this.input] }),
      });
      const data = await res.json();
      this.response = `Documents indexés : ${data.indexed}`;
    },
    async search() {
      const res = await fetch("http://localhost:5000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: this.input }),
      });
      const data = await res.json();
      this.response = data.results.join("\n");
    },
    async askAgent() {
      const res = await fetch("http://localhost:5000/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: this.input }),
      });
      const data = await res.json();
      this.response = data.answer;
    },
  },
};
</script>

<style scoped>
.container {
  padding: 2rem;
  max-width: 600px;
  margin: auto;
}
textarea {
  width: 100%;
  height: 120px;
  margin-bottom: 1rem;
}
.buttons button {
  margin-right: 10px;
}
pre {
  background: #111;
  padding: 1rem;
  border-radius: 5px;
}
</style>
