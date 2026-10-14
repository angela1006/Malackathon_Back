from fastapi import FastAPI

app = FastAPI(title="Backend Hackaton")

@app.get("/")
def inicio():
    return {"mensaje": "¡Backend funcionando correctamente!"}

@app.get("/clientes")
def obtener_clientes():
    return {
        "status": "ok",
        "data": [
            {"id": 1, "nombre": "Cliente A", "gasto": 45000},
            {"id": 2, "nombre": "Cliente B", "gasto": 12000}
        ]
    }