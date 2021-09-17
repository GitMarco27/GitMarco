from GitMarco.energy.solar import DirectRadiation


def test_direct_radiation():
    inputs = {
        'latitude': 23,
        'longitude': 13,
        'nmin': 60 * 24,
        'start_year': 2015,
        'start_day': 27,
        'start_month': 10,
        'timezone': 1
    }
    model = DirectRadiation(latitude=inputs['latitude'],
                            longitude=inputs['longitude'],
                            nmin=inputs['nmin'],
                            start_day=inputs['start_day'],
                            start_year=inputs['start_year'],
                            start_month=inputs['start_month'],
                            timezone=inputs['timezone']
                            )

    model.estimate_direct_radiation()
