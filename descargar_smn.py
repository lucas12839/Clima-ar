print()
print("======================================")
print("       ClimaAR - Datos horarios SMN")
print("======================================")
print(f"Desde: {desde}")
print(f"Hasta: {hasta}")
print(f"Salida: {args.salida}")
print()
print("Estaciones:")

for estacion in sorted(ESTACIONES):
    print(f"  - {estacion}")

print()

while fecha_actual <= hasta:

    registros, error = descargar_dia(fecha_actual)

    if error:
        if error == "archivo_vacio":
            dias_sin_datos += 1
            print(f"[SIN DATOS] {fecha_actual}")
        else:
            errores += 1
            print(f"[ERROR] {fecha_actual} | {error}")

    elif registros:
        nuevos = guardar_registros(
            args.salida,
            registros,
            existentes,
        )

        dias_ok += 1
        total_nuevos += nuevos

        print(
            f"[OK] {fecha_actual} | "
            f"{len(registros)} registros encontrados | "
            f"{nuevos} nuevos"
        )

    else:
        dias_sin_datos += 1
        print(f"[SIN DATOS] {fecha_actual}")

    fecha_actual += timedelta(days=1)

    if fecha_actual <= hasta:
        time.sleep(args.pausa)

print()
print("======================================")
print("               RESUMEN")
print("======================================")
print(f"Días con datos: {dias_ok}")
print(f"Días sin datos: {dias_sin_datos}")
print(f"Errores: {errores}")
print(f"Registros nuevos: {total_nuevos}")
print(f"Salida: {os.path.abspath(args.salida)}")
print("======================================")


if __name__ == "__main__":
    main()
